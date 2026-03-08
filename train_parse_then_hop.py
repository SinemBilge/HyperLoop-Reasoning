import argparse
from typing import Union
from math import exp, log

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from src.utils.util import set_seed
from src.config import Config
from src.datasets import get_parse_dataset
from src.models import T5ModelWithAdditionalLayer, SoftPromptModel
from src.train.soft_prompt_trainer import SoftPromptTrainer

config = Config()


def _train_parse_then_hop(
    additional_layer: str,
    dataset: str,
    rank: int,
    world_size: int,
    lr: float = 0.3,
    curvature: float = 1.0,
    knit5_checkpoint_path: Union[str, None] = None,
    checkpoint_save_path: Union[str, None] = None,
    tboard_logs_save_path: Union[str, None] = None,
    epochs: Union[int, None] = None,
    batch_size: Union[int, None] = None,
    additional_layer_lr: float = 0.001,
    num_loop_iterations: int = 1,
):
    gpu_parallel = config.parse_then_hop_training.gpu_parallelization

    parse_train, parse_dev, _ = get_parse_dataset(dataset)
    print(f"Number of Parse Questions Train: {len(parse_train)}")
    print(f"Number of Parse Questions Dev: {len(parse_dev)}")

    # device
    if torch.cuda.is_available():
        device = f"cuda:{rank}" if gpu_parallel else "cuda"
    else:
        device = "cpu"

    print(f"Process rank: {rank} using device: {device}")
    if dist.is_initialized():
        print(f"Rank: {dist.get_rank()} / World size: {dist.get_world_size()}")

    if batch_size is not None:
        config.t5_model.batch_size = batch_size
        print(f"Setting batch size to: {batch_size}")

    # DataLoaders
    if gpu_parallel:
        parse_sampler_train = DistributedSampler(
            parse_train, shuffle=True, num_replicas=world_size, rank=rank
        )
        parse_sampler_dev = DistributedSampler(
            parse_dev, shuffle=False, num_replicas=world_size, rank=rank
        )
        per_gpu_batch = max(1, config.t5_model.batch_size // world_size)
        parse_dataloader_train = DataLoader(
            parse_train,
            sampler=parse_sampler_train,
            batch_size=per_gpu_batch,
            num_workers=config.parse_then_hop_training.num_workers,
        )
        parse_dataloader_dev = DataLoader(
            parse_dev,
            sampler=parse_sampler_dev,
            batch_size=per_gpu_batch,
            num_workers=config.parse_then_hop_training.num_workers,
        )
    else:
        parse_dataloader_train = DataLoader(
            parse_train,
            batch_size=config.t5_model.batch_size,
            shuffle=True,
            num_workers=config.parse_then_hop_training.num_workers,
        )
        parse_dataloader_dev = DataLoader(
            parse_dev,
            batch_size=config.t5_model.batch_size,
            shuffle=False,
            num_workers=config.parse_then_hop_training.num_workers,
        )

    # Tokenizer & configs
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config.parse_then_hop_training.learning_rate = lr
    print(f"Setting parsing LR (soft prompt) to {lr}")
    config.parse_then_hop_training.curvature = log(exp(curvature) - 1.0)
    print(f"Setting Curvature to {curvature}")
    config.parse_then_hop_training.model_checkpoint_path = knit5_checkpoint_path
    print(f"Setting KNIT5 Checkpoint Load Path to: {knit5_checkpoint_path}")

    config.single_hop_training.learning_rate = additional_layer_lr
    print(f"Setting additional layer learning rate to {additional_layer_lr}")

    # Model
    print("Loading KNIT5 model with additional layer...")
    config.parse_then_hop_training.num_loop_iterations = num_loop_iterations
    print(f"Setting num_loop_iterations to {num_loop_iterations}")

    hyperbolic_knit5_model = T5ModelWithAdditionalLayer(
        layer_type=additional_layer,
        curvature=config.parse_then_hop_training.curvature,
        checkpoint_hyperbolic_knit5=config.parse_then_hop_training.model_checkpoint_path,
        with_model_state_dict=True,
        num_loop_iterations=num_loop_iterations,
    )
    model = SoftPromptModel(
        knit5=hyperbolic_knit5_model, model_name="parsing_prompt"
    )
    print(
        f"Train hyperbolic Soft Prompt Model with additional layer {additional_layer} "
        f"and curvature {config.parse_then_hop_training.curvature if additional_layer == 'hyperbolic' else 0.0}"
    )

    if epochs is not None:
        config.parse_then_hop_training.epochs = epochs
        print(f"Setting epochs to: {epochs}")

    if checkpoint_save_path is not None:
        config.parse_then_hop_training.model_save_path = checkpoint_save_path
        print(f"Setting Checkpoint Save path to: {checkpoint_save_path}")
    if tboard_logs_save_path is not None:
        config.parse_then_hop_training.log_dir = tboard_logs_save_path
        print(f"Setting Tensorboard Log save path to: {tboard_logs_save_path}")

    config.parse_then_hop_training.additional_log_info = (
        f"{additional_layer}_after_encoder_bsize{config.t5_model.batch_size}"
        f"_prompt_lenght{config.parse_then_hop_training.prompt_length}"
        f"_lr{config.parse_then_hop_training.learning_rate}"
        f"_curvature{curvature}_additional_layer_lr{additional_layer_lr}"
        f"_loop{num_loop_iterations}"
    )

    trainer = SoftPromptTrainer(
        model,
        tokenizer,
        parse_dataloader_train,
        parse_dataloader_dev,
        config,
        device=device,
        method="parse_then_hop_training",
        checkpoint_path=config.parse_then_hop_training.hopping_prompt_checkpoint_path,
        tboard_checkpoint_path=config.parse_then_hop_training.tboard_checkpoint_path,
        retrain=True,
        gpu_parallelization=gpu_parallel,
        rank=rank,
    )

    print("Parsing (parse-then-hop) training..")
    print(f"Base model: {config.t5_model.model_name}")
    print(f"Epochs: {config.parse_then_hop_training.epochs}")
    print(f"Global batch size: {config.t5_model.batch_size}")
    print(f"Optimizer: {config.parse_then_hop_training.optimizer}")

    trainer.train(epochs=config.parse_then_hop_training.epochs)


def setup_ddp(rank: int, world_size: int):
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    torch.cuda.set_device(rank)
    set_seed(42)


def train_ddp(
    rank: int,
    world_size: int,
    dataset: str,
    additional_layer: str,
    lr: float,
    curvature: float,
    knit5_checkpoint_path: Union[str, None],
    checkpoint_save_path: Union[str, None],
    tboard_logs_save_path: Union[str, None],
    epochs: int,
    batch_size: int,
    additional_layer_lr: float,
    num_loop_iterations: int,
):
    setup_ddp(rank, world_size)
    try:
        _train_parse_then_hop(
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
            num_loop_iterations=num_loop_iterations,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Then Hop Training")
    parser.add_argument("--hyperbolic", action="store_true")
    parser.add_argument(
        "--additional_layer",
        type=str,
        choices=["identity", "hyperbolic", "euclidean"],
        default="identity",
    )
    parser.add_argument("--learning_rate", type=float, default=0.3)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--knit5_checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_save_path", type=str, default=None)
    parser.add_argument("--tboard_logs_save_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--additional_layer_lr", type=float, default=0.001
    )
    parser.add_argument("--dataset", type=str, nargs="?", default=None)
    parser.add_argument("--num_loop_iterations", type=int, default=1)
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
    num_loop_iterations = args.num_loop_iterations

    gpu_parallel = config.parse_then_hop_training.gpu_parallelization
    if gpu_parallel and world_size > 1:
        mp.spawn(
            train_ddp,
            args=(
                world_size,
                dataset,
                additional_layer,
                lr,
                curvature,
                knit5_checkpoint_path,
                checkpoint_save_path,
                tboard_logs_save_path,
                epochs,
                batch_size,
                additional_layer_lr,
                num_loop_iterations,
            ),
            nprocs=world_size,
            join=True,
        )
    else:
        _train_parse_then_hop(
            additional_layer=additional_layer,
            dataset=dataset,
            rank=0,
            world_size=1,
            lr=lr,
            curvature=curvature,
            knit5_checkpoint_path=knit5_checkpoint_path,
            checkpoint_save_path=checkpoint_save_path,
            tboard_logs_save_path=tboard_logs_save_path,
            epochs=epochs,
            batch_size=batch_size,
            additional_layer_lr=additional_layer_lr,
            num_loop_iterations=num_loop_iterations,
        )
