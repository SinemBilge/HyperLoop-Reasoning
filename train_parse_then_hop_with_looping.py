"""
Train Parse Then Hop WITH LOOPING

Same as train_parse_then_hop.py but uses T5ModelWithLoopedHyperbolic
which loops the hyperbolic layer during training.

Usage:
    python train_parse_then_hop_with_looping.py \
        --additional_layer hyperbolic \
        --dataset wikimultihop \
        --knit5_checkpoint_path <path> \
        --num_loops 4 \
        --max_norm 0.9 \
        --epochs 50
"""

from src.utils.util import load_dataset, load_train_test_pql_dataset
from src.train import *
import pandas as pd
from src.knowledge_graph import create_knowledge_graph_wikimultihop
from src.datasets import ParseMetaQADataset, ParseMLPQDataset, ParsePQLDataset, ParseWikHopDataset

from src.config import Config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from src.models import SoftPromptModel
from src.models.hyperbolic_t5_with_looping import T5ModelWithLoopedHyperbolic
import argparse

import torch.distributed as dist
import torch.multiprocessing as mp
import os
from src.utils.util import set_seed
from math import exp, log
from src.datasets import get_parse_dataset

config = Config()


def _train_parse_then_hop_with_looping(
        additional_layer: str,
        dataset: str,
        rank,
        world_size,
        lr=0.3,
        curvature=1.0,
        knit5_checkpoint_path=None,
        checkpoint_save_path=None,
        tboard_logs_save_path=None,
        epochs=None,
        batch_size=None,
        additional_layer_lr=0.001,
        num_loops=4,
        max_norm=0.9,
        input_injection=True):

    GPU_PARALLELIZATION = True
    WITH_MODEL_STATE_DICT = GPU_PARALLELIZATION
    parse_train, parse_dev, _ = get_parse_dataset(dataset)
    print(f"Number of Parse Questions Train: {len(parse_train)}")
    print(f"Number of Parse Questions Dev: {len(parse_dev)}")

    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    print(f"Process rank: {rank} using device: {device}")
    if dist.is_initialized():
        print(f"Rank: {dist.get_rank()} / World size: {dist.get_world_size()}")
    if batch_size is not None:
        config.t5_model.batch_size = batch_size
        print(f"Setting batch size to: {batch_size}")

    if config.parse_then_hop_training.gpu_parallelization:
        from torch.utils.data import DistributedSampler

        parse_sampler_train = DistributedSampler(parse_train, shuffle=True, num_replicas=world_size, rank=rank)
        parse_sampler_dev = DistributedSampler(parse_dev, shuffle=False, num_replicas=world_size, rank=rank)

        parse_dataloader_train = DataLoader(parse_train, sampler=parse_sampler_train, batch_size=config.t5_model.batch_size//world_size, num_workers=config.parse_then_hop_training.num_workers)
        parse_dataloader_dev = DataLoader(parse_dev, sampler=parse_sampler_dev, batch_size=config.t5_model.batch_size//world_size, num_workers=config.parse_then_hop_training.num_workers)
    else:
        parse_dataloader_train = DataLoader(parse_train, batch_size=config.t5_model.batch_size, shuffle=True, num_workers=config.parse_then_hop_training.num_workers)
        parse_dataloader_dev = DataLoader(parse_dev, batch_size=config.t5_model.batch_size, shuffle=False, num_workers=config.parse_then_hop_training.num_workers)

    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config.parse_then_hop_training.learning_rate = lr
    print(f"Setting learning rate to {lr}")
    config.parse_then_hop_training.curvature = log(exp(curvature) - 1)
    print(f"Setting Curvature to {curvature}")
    config.parse_then_hop_training.model_checkpoint_path = knit5_checkpoint_path
    print(f"Setting KNIT5 Checkpoint Load Path to: {knit5_checkpoint_path}")
    config.single_hop_training.learning_rate = additional_layer_lr
    print(f"Setting additional layer learning rate to {additional_layer_lr}")

    print("Loading Model WITH LOOPING...")
    print(f"  num_loops={num_loops}, max_norm={max_norm}, input_injection={input_injection}")

    # Use the new looped model instead of T5ModelWithAdditionalLayer
    hyperbolic_knit5_model = T5ModelWithLoopedHyperbolic(
        layer_type=additional_layer,
        curvature=config.parse_then_hop_training.curvature,
        checkpoint_hyperbolic_knit5=config.parse_then_hop_training.model_checkpoint_path,
        with_model_state_dict=WITH_MODEL_STATE_DICT,
        gpu_parallelization=GPU_PARALLELIZATION,
        soft_prompt_length=config.parse_then_hop_training.prompt_length,
        num_loops=num_loops,
        max_norm=max_norm,
        input_injection=input_injection
    )

    model = SoftPromptModel(knit5=hyperbolic_knit5_model, model_name='parsing_prompt_looped')
    print(f"Train with LOOPED hyperbolic Soft Prompt Model")
    print(f"  additional layer: {additional_layer}")
    print(f"  curvature: {config.parse_then_hop_training.curvature if additional_layer == 'hyperbolic' else 0.0}")
    print(f"  num_loops: {num_loops}")

    if epochs is not None:
        config.parse_then_hop_training.epochs = epochs
        print(f"Setting epochs to: {epochs}")

    if checkpoint_save_path is not None:
        config.parse_then_hop_training.model_save_path = checkpoint_save_path
        print(f"Setting Checkpoint Save path to: {checkpoint_save_path}")
    if tboard_logs_save_path is not None:
        config.parse_then_hop_training.log_dir = tboard_logs_save_path
        print(f"Setting Tensorboard Log save path to: {tboard_logs_save_path}")

    config.parse_then_hop_training.additional_log_info = f'{additional_layer}_LOOPED{num_loops}_maxnorm{max_norm}_bsize{config.t5_model.batch_size}_prompt_length{config.parse_then_hop_training.prompt_length}_lr{config.parse_then_hop_training.learning_rate}_curvature{curvature}'

    trainer = SoftPromptTrainer(
        model,
        tokenizer,
        parse_dataloader_train,
        parse_dataloader_dev,
        config,
        device=device,
        method='parse_then_hop_training',
        checkpoint_path=config.parse_then_hop_training.hopping_prompt_checkpoint_path,
        tboard_checkpoint_path=config.parse_then_hop_training.tboard_checkpoint_path,
        retrain=True,
        gpu_parallelization=config.parse_then_hop_training.gpu_parallelization,
        rank=rank
    )

    print(f'Parsing WITH LOOPING...')
    print(f'with model: {config.t5_model.model_name}')
    print(f'Model Config: {model.knit5.config}')
    print(f'for: {config.parse_then_hop_training.epochs} epochs')
    print(f'with batch size: {config.t5_model.batch_size}')
    print(f'with optimizer: {config.parse_then_hop_training.optimizer}')

    trainer.train(epochs=config.parse_then_hop_training.epochs)


def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)
    set_seed(42)


def train_ddp(rank, world_size, dataset, additional_layer, lr, curvature, knit5_checkpoint_path,
              checkpoint_save_path, tboard_logs_save_path, epochs, batch_size, additional_layer_lr,
              num_loops, max_norm, input_injection):
    setup_ddp(rank, world_size)
    _train_parse_then_hop_with_looping(
        additional_layer=additional_layer,
        dataset=dataset,
        rank=rank,
        world_size=world_size,
        lr=lr,
        curvature=curvature,
        knit5_checkpoint_path=knit5_checkpoint_path,
        checkpoint_save_path=checkpoint_save_path,
        tboard_logs_save_path=tboard_logs_save_path,
        epochs=epochs,
        batch_size=batch_size,
        additional_layer_lr=additional_layer_lr,
        num_loops=num_loops,
        max_norm=max_norm,
        input_injection=input_injection
    )
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse Then Hop Training WITH LOOPING')

    parser.add_argument('--hyperbolic', action='store_true', help='Train with hyperbolic representation')
    parser.add_argument(
        '--additional_layer',
        type=str,
        choices=['identity', 'hyperbolic', 'linear'],
        default='hyperbolic',
        help='Specify the type of additional layer to use: identity, hyperbolic, or linear'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.3,
        help='Specify the Learning Rate'
    )
    parser.add_argument(
        '--curvature',
        type=float,
        default=1.0,
        help='Specify curvature for Hyperbolic Layer'
    )
    parser.add_argument(
        '--knit5_checkpoint_path',
        type=str,
        default=None,
        help='Specify Checkpoint Path of finetuned KNIT5 Model'
    )
    parser.add_argument(
        '--checkpoint_save_path',
        type=str,
        default=None,
        help='Specify save path for the checkpoint'
    )
    parser.add_argument(
        '--tboard_logs_save_path',
        type=str,
        default=None,
        help='Specify path for tensorboard logs'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Specify number of epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Specify batch size'
    )
    parser.add_argument(
        '--additional_layer_lr',
        type=float,
        default=0.001,
        help='Specify learning rate for additional layer'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        nargs='?',
        default=None,
        help='Specify the dataset (e.g., metaqa, 2wikimultihop)'
    )

    # NEW: Looping parameters
    parser.add_argument(
        '--num_loops',
        type=int,
        default=4,
        help='Number of times to loop the hyperbolic layer'
    )
    parser.add_argument(
        '--max_norm',
        type=float,
        default=0.9,
        help='Max norm for clamping (keeps values inside Poincare ball)'
    )
    parser.add_argument(
        '--no_input_injection',
        action='store_true',
        help='Disable input injection (not recommended)'
    )

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    additional_layer = args.additional_layer
    lr = args.learning_rate
    curvature = args.curvature
    knit5_checkpoint_path = args.knit5_checkpoint_path
    checkpoint_save_path = args.checkpoint_save_path
    tboard_logs_save_path = args.tboard_logs_save_path
    epochs = args.epochs
    dataset = args.dataset
    batch_size = args.batch_size
    additional_layer_lr = args.additional_layer_lr
    num_loops = args.num_loops
    max_norm = args.max_norm
    input_injection = not args.no_input_injection

    mp.spawn(
        train_ddp,
        args=(world_size, dataset, additional_layer, lr, curvature, knit5_checkpoint_path,
              checkpoint_save_path, tboard_logs_save_path, epochs, batch_size, additional_layer_lr,
              num_loops, max_norm, input_injection),
        nprocs=world_size,
        join=True
    )
