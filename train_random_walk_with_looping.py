"""
Train Random Walk (Hopping) WITH LOOPING

Same as train_random_walk.py but uses T5ModelWithLoopedHyperbolic
which loops the hyperbolic layer during training.

Usage:
    python train_random_walk_with_looping.py \
        --additional_layer hyperbolic \
        --dataset wikimultihop \
        --knit5_checkpoint_path <path> \
        --num_loops 4 \
        --max_norm 0.9 \
        --epochs 50
"""

from src.utils.util import load_dataset, get_top_token_embeddings, load_train_test_pql_dataset
import pandas as pd
from src.train.soft_prompt_trainer import SoftPromptTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from src.config import Config
from torch.utils.data import DataLoader
from src.knowledge_graph import create_knowledge_graph_wikimultihop, create_knowledge_graph_metaqa, create_knowledge_graph_mlpq, create_knowledge_graph_pql
from src.models import SoftPromptModel
from src.models.hyperbolic_t5_with_looping import T5ModelWithLoopedHyperbolic
from src.datasets import RandomWalkMetaQADataset, RandomWalkMLPQDataset, RandomWalkWikiHopDataset, RandomWalkPQLDataset
import argparse
import os
from math import exp, log
from src.datasets.dataloader import get_random_walk_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
config = Config()

import torch.distributed as dist
import torch.multiprocessing as mp
from src.utils.util import set_seed


def _train_random_walk_with_looping(
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
        batch_size=128,
        additional_layer_lr=0.001,
        no_soft_prompt=False,
        use_scheduler=False,
        checkpoint_load_path=None,
        tboard_logs_load_path=None,
        num_loops=4,
        max_norm=0.9,
        input_injection=True):

    GPU_PARALLELIZATION = True
    WITH_MODEL_STATE_DICT = GPU_PARALLELIZATION
    random_walk_train, random_walk_dev, _ = get_random_walk_dataset(dataset)
    print(f"Number of Random Walks Train: {len(random_walk_train)}")
    print(f"Number of Random Walk Dev: {len(random_walk_dev)}")

    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    print(f"Process rank: {rank} using device: {device}")
    if dist.is_initialized():
        print(f"Rank: {dist.get_rank()} / World size: {dist.get_world_size()}")

    config.t5_model.batch_size = batch_size
    print(f"Setting batch_size to: {batch_size}")

    if config.random_walk_training.gpu_parallelization:
        from torch.utils.data import DistributedSampler

        random_walk_sampler_train = DistributedSampler(random_walk_train, shuffle=True, num_replicas=world_size, rank=rank)
        random_walk_sampler_dev = DistributedSampler(random_walk_dev, shuffle=False, num_replicas=world_size, rank=rank)

        random_walk_dataloader_train = DataLoader(random_walk_train, sampler=random_walk_sampler_train, batch_size=config.t5_model.batch_size//world_size, num_workers=config.random_walk_training.num_workers)
        random_walk_dataloader_dev = DataLoader(random_walk_dev, sampler=random_walk_sampler_dev, batch_size=config.t5_model.batch_size//world_size, num_workers=config.random_walk_training.num_workers)
    else:
        random_walk_dataloader_train = DataLoader(random_walk_train, batch_size=config.t5_model.batch_size, shuffle=True, num_workers=config.random_walk_training.num_workers)
        random_walk_dataloader_dev = DataLoader(random_walk_dev, batch_size=config.t5_model.batch_size, shuffle=False, num_workers=config.random_walk_training.num_workers)

    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Find longest sequence
    longest_sequence = 0
    for incomplete, complete in random_walk_dev:
        tokens = tokenizer(complete, return_tensors='pt', truncation=False).input_ids
        token_length = tokens.size(1)
        if token_length > longest_sequence:
            longest_sequence = token_length
    for incomplete, complete in random_walk_train:
        tokens = tokenizer(complete, return_tensors='pt', truncation=False).input_ids
        token_length = tokens.size(1)
        if token_length > longest_sequence:
            longest_sequence = token_length
    print(f"Longest Sequence has {longest_sequence} tokens")

    config.random_walk_training.use_soft_prompt = not no_soft_prompt
    print(f"Using Soft Prompt: {not no_soft_prompt}")

    tokenizer.model_max_length = config.t5_model.tokenizer_max_length
    config.random_walk_training.learning_rate = lr
    print(f"Setting learning rate to {lr}")
    config.random_walk_training.curvature = log(exp(curvature) - 1)
    print(f"Setting Curvature to {curvature}")
    config.single_hop_training.learning_rate = additional_layer_lr
    print(f"Setting additional layer learning rate to {additional_layer_lr}")
    config.random_walk_training.use_scheduler = use_scheduler
    print(f"Setting use scheduler to {use_scheduler}")
    config.random_walk_training.hopping_prompt_checkpoint_path = checkpoint_load_path
    print(f"Using {checkpoint_load_path} to Load.")
    config.random_walk_training.tboard_checkpoint_path = tboard_logs_load_path
    print(f"Using {tboard_logs_load_path} for Logs.")

    print("Loading Model WITH LOOPING...")
    print(f"  num_loops={num_loops}, max_norm={max_norm}, input_injection={input_injection}")

    config.random_walk_training.model_checkpoint_path = knit5_checkpoint_path
    print(f"Setting KNIT5 Checkpoint Load Path to: {knit5_checkpoint_path}")

    # Use the new looped model instead of T5ModelWithAdditionalLayer
    hyperbolic_knit5_model = T5ModelWithLoopedHyperbolic(
        layer_type=additional_layer,
        curvature=config.random_walk_training.curvature,
        checkpoint_hyperbolic_knit5=config.random_walk_training.model_checkpoint_path,
        with_model_state_dict=WITH_MODEL_STATE_DICT,
        gpu_parallelization=GPU_PARALLELIZATION,
        soft_prompt_length=config.random_walk_training.prompt_length,
        num_loops=num_loops,
        max_norm=max_norm,
        input_injection=input_injection
    )

    model = SoftPromptModel(
        knit5=hyperbolic_knit5_model,
        model_name='hyperbolic_hopping_prompt_looped',
        soft_prompt_length=config.random_walk_training.prompt_length
    )

    print(f"Train with LOOPED hyperbolic Soft Prompt Model")
    print(f"  additional layer: {additional_layer}")
    print(f"  curvature: {config.random_walk_training.curvature if additional_layer == 'hyperbolic' else 0.0}")
    print(f"  num_loops: {num_loops}")

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {num_trainable_params}")

    if epochs is not None:
        config.random_walk_training.epochs = epochs
        print(f"Setting epochs to: {epochs}")

    if checkpoint_save_path is not None:
        config.random_walk_training.model_save_path = checkpoint_save_path
        print(f"Setting Checkpoint Save path to: {checkpoint_save_path}")
    if tboard_logs_save_path is not None:
        config.random_walk_training.log_dir = tboard_logs_save_path
        print(f"Setting Tensorboard Log save path to: {tboard_logs_save_path}")

    config.random_walk_training.additional_log_info = f'{additional_layer}_LOOPED{num_loops}_maxnorm{max_norm}_bsize{config.t5_model.batch_size}_prompt_length{config.random_walk_training.prompt_length}_lr{config.random_walk_training.learning_rate}_curvature{curvature}_use_prompt_{config.random_walk_training.use_soft_prompt}'

    trainer = SoftPromptTrainer(
        model,
        tokenizer,
        random_walk_dataloader_train,
        random_walk_dataloader_dev,
        config,
        device=device,
        method='random_walk_training',
        checkpoint_path=config.random_walk_training.hopping_prompt_checkpoint_path,
        tboard_checkpoint_path=config.random_walk_training.tboard_checkpoint_path,
        retrain=True,
        gpu_parallelization=config.random_walk_training.gpu_parallelization,
        rank=rank
    )

    print(f'Random Walk Training WITH LOOPING...')
    print(f'with model: {config.t5_model.model_name}')
    print(f'with lr: {config.random_walk_training.learning_rate}')
    print(f'for: {config.random_walk_training.epochs} epochs')
    print(f'with effective batch size: {config.t5_model.batch_size} ({config.t5_model.batch_size / world_size} per GPU)')
    print(f'with optimizer: {config.random_walk_training.optimizer}')

    trainer.train(epochs=config.random_walk_training.epochs)


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
              no_soft_prompt, use_scheduler, checkpoint_load_path, tboard_logs_load_path,
              num_loops, max_norm, input_injection):
    setup_ddp(rank, world_size)
    _train_random_walk_with_looping(
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
        no_soft_prompt=no_soft_prompt,
        use_scheduler=use_scheduler,
        checkpoint_load_path=checkpoint_load_path,
        tboard_logs_load_path=tboard_logs_load_path,
        num_loops=num_loops,
        max_norm=max_norm,
        input_injection=input_injection
    )
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Walk Training WITH LOOPING')

    parser.add_argument('--dataset', type=str, nargs='?', default=None, help='Specify the dataset')
    parser.add_argument(
        '--additional_layer',
        type=str,
        choices=['identity', 'hyperbolic', 'linear'],
        default='hyperbolic',
        help='Specify the type of additional layer'
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
        '--epochs',
        type=int,
        default=100,
        help='Specify number of epochs'
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
        '--checkpoint_load_path',
        type=str,
        default=None,
        help='Specify path for Load Checkpoint'
    )
    parser.add_argument(
        '--tboard_logs_load_path',
        type=str,
        default=None,
        help='Specify path for Load tensorboard logs'
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
        '--no_soft_prompt',
        action='store_true',
        help='If set, dont use soft prompt',
        default=False
    )
    parser.add_argument(
        '--use_scheduler',
        action='store_true',
        help='If set, use scheduler'
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
    dataset = args.dataset
    additional_layer = args.additional_layer
    lr = args.learning_rate
    curvature = args.curvature
    knit5_checkpoint_path = args.knit5_checkpoint_path
    checkpoint_save_path = args.checkpoint_save_path
    tboard_logs_save_path = args.tboard_logs_save_path
    epochs = args.epochs
    batch_size = args.batch_size
    additional_layer_lr = args.additional_layer_lr
    no_soft_prompt = args.no_soft_prompt
    use_scheduler = args.use_scheduler
    checkpoint_load_path = args.checkpoint_load_path
    tboard_logs_load_path = args.tboard_logs_load_path
    num_loops = args.num_loops
    max_norm = args.max_norm
    input_injection = not args.no_input_injection

    mp.spawn(
        train_ddp,
        args=(world_size, dataset, additional_layer, lr, curvature, knit5_checkpoint_path,
              checkpoint_save_path, tboard_logs_save_path, epochs, batch_size, additional_layer_lr,
              no_soft_prompt, use_scheduler, checkpoint_load_path, tboard_logs_load_path,
              num_loops, max_norm, input_injection),
        nprocs=world_size,
        join=True
    )
