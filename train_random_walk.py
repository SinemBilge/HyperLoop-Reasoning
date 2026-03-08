import argparse
import os
from math import exp, log
from typing import Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from src.utils.util import set_seed
from src.config import Config
from src.datasets.dataloader import get_random_walk_dataset
from src.models import SoftPromptModel, T5ModelWithAdditionalLayer
from src.train.soft_prompt_trainer import SoftPromptTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
config = Config()


def _train_random_walk(
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
    batch_size: int = 128,
    additional_layer_lr: float = 0.001,
    no_soft_prompt: bool = False,
    use_scheduler: bool = False,
    num_layers: int = 1,
    checkpoint_load_path: Union[str, None] = None,
    tboard_logs_load_path: Union[str, None] = None,
    num_loop_iterations: int = 1,
):
    gpu_parallel = config.random_walk_training.gpu_parallelization

    random_walk_train, random_walk_dev, _ = get_random_walk_dataset(dataset)
    print(f"Number of Random Walks Train: {len(random_walk_train)}")
    print(f"Number of Random Walk Dev: {len(random_walk_dev)}")

    # device
    if torch.cuda.is_available():
        device = f"cuda:{rank}" if gpu_parallel else "cuda"
    else:
        device = "cpu"

    print(f"Process rank: {rank} using device: {device}")
    if dist.is_initialized():
        print(f"Rank: {dist.get_rank()} / World size: {dist.get_world_size()}")

    config.t5_model.batch_size = batch_size
    print(f"Setting batch_size to: {batch_size}")

    # DataLoaders
    if gpu_parallel:
        random_walk_sampler_train = DistributedSampler(
            random_walk_train, shuffle=True, num_replicas=world_size, rank=rank
        )
        random_walk_sampler_dev = DistributedSampler(
            random_walk_dev, shuffle=False, num_replicas=world_size, rank=rank
        )
        per_gpu_batch = max(1, config.t5_model.batch_size // world_size)
        random_walk_dataloader_train = DataLoader(
            random_walk_train,
            sampler=random_walk_sampler_train,
            batch_size=per_gpu_batch,
            num_workers=config.random_walk_training.num_workers,
        )
        random_walk_dataloader_dev = DataLoader(
            random_walk_dev,
            sampler=random_walk_sampler_dev,
            batch_size=per_gpu_batch,
            num_workers=config.random_walk_training.num_workers,
        )
    else:
        random_walk_dataloader_train = DataLoader(
            random_walk_train,
            batch_size=config.t5_model.batch_size,
            shuffle=True,
            num_workers=config.random_walk_training.num_workers,
        )
        random_walk_dataloader_dev = DataLoader(
            random_walk_dev,
            batch_size=config.t5_model.batch_size,
            shuffle=False,
            num_workers=config.random_walk_training.num_workers,
        )

    # Tokenizer & configs
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Just to get an idea of sequence length (kept from your script)
    longest_sequence = 0
    for _, complete in random_walk_dev:
        tokens = tokenizer(complete, return_tensors="pt", truncation=False).input_ids
        longest_sequence = max(longest_sequence, tokens.size(1))
    for _, complete in random_walk_train:
        tokens = tokenizer(complete, return_tensors="pt", truncation=False).input_ids
        longest_sequence = max(longest_sequence, tokens.size(1))

    config.random_walk_training.use_soft_prompt = not no_soft_prompt
    print(f"Using Soft Prompt: {config.random_walk_training.use_soft_prompt}")

    tokenizer.model_max_length = config.t5_model.tokenizer_max_length
    config.random_walk_training.learning_rate = lr
    print(f"Setting learning rate to {lr}")
    config.random_walk_training.curvature = log(exp(curvature) - 1.0)
    print(f"Setting Curvature to {curvature}")
    config.single_hop_training.learning_rate = additional_layer_lr
    print(f"Setting additional layer learning rate to {additional_layer_lr}")
    config.random_walk_training.use_scheduler = use_scheduler
    print(f"Setting use scheduler to {use_scheduler}")
    config.random_walk_training.hopping_prompt_checkpoint_path = checkpoint_load_path
    print(f"Using {checkpoint_load_path} Checkpoint to Load Soft Prompt.")
    config.random_walk_training.tboard_checkpoint_path = tboard_logs_load_path
    print(f"Using {tboard_logs_load_path} for Logs.")

    print("Loading KNIT5 model with additional layer...")
    config.random_walk_training.model_checkpoint_path = knit5_checkpoint_path
    print(f"Setting KNIT5 Checkpoint Load Path to: {knit5_checkpoint_path}")
    print(f"Number of Layers are {num_layers}")
    config.random_walk_training.num_loop_iterations = num_loop_iterations
    print(f"Setting num_loop_iterations to {num_loop_iterations}")

    hyperbolic_knit5_model = T5ModelWithAdditionalLayer(
        layer_type=additional_layer,
        num_layers=num_layers,
        curvature=config.random_walk_training.curvature,
        checkpoint_hyperbolic_knit5=config.random_walk_training.model_checkpoint_path,
        with_model_state_dict=True,
        num_loop_iterations=num_loop_iterations,
    )
    model = SoftPromptModel(
        knit5=hyperbolic_knit5_model,
        model_name="hyperbolic_hopping_prompt",
        soft_prompt_length=config.random_walk_training.prompt_length,
    )
    print(
        f"Train Soft Prompt Model with additional layer {additional_layer} "
        f"and curvature {config.random_walk_training.curvature if additional_layer == 'hyperbolic' else 0.0}"
    )

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

    c_val = 0.0
    if additional_layer == "hyperbolic" and hasattr(
        model.knit5.additional_layer, "manifold"
    ):
        c_val = model.knit5.additional_layer.manifold.c.item()

    config.random_walk_training.additional_log_info = (
        f"{additional_layer}_after_encoder_bsize{config.t5_model.batch_size}"
        f"_prompt_lenght{config.random_walk_training.prompt_length}"
        f"_lr{config.random_walk_training.learning_rate}"
        f"_curvature{c_val}_additional_layer_lr{additional_layer_lr}"
        f"_use_prompt_{config.random_walk_training.use_soft_prompt}"
        f"_loop{num_loop_iterations}"
        f"{'_cont' if config.random_walk_training.hopping_prompt_checkpoint_path else ''}"
    )

    trainer = SoftPromptTrainer(
        model,
        tokenizer,
        random_walk_dataloader_train,
        random_walk_dataloader_dev,
        config,
        device=device,
        method="random_walk_training",
        checkpoint_path=config.random_walk_training.hopping_prompt_checkpoint_path,
        tboard_checkpoint_path=config.random_walk_training.tboard_checkpoint_path,
        retrain=True,
        gpu_parallelization=gpu_parallel,
        rank=rank,
    )

    print("Random Walk Training..")
    print(f"Base model: {config.t5_model.model_name}")
    print(f"LR: {config.random_walk_training.learning_rate}")
    print(
        f"Epochs: {config.random_walk_training.epochs}, "
        f"effective global batch size: {config.t5_model.batch_size} "
        f"({config.t5_model.batch_size / world_size} per GPU)"
    )
    print(f"Optimizer: {config.random_walk_training.optimizer}")

    trainer.train(epochs=config.random_walk_training.epochs)


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
    no_soft_prompt: bool,
    use_scheduler: bool,
    num_layers: int,
    checkpoint_load_path: Union[str, None],
    tboard_logs_load_path: Union[str, None],
    num_loop_iterations: int,
):
    setup_ddp(rank, world_size)
    try:
        _train_random_walk(
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
            num_layers=num_layers,
            checkpoint_load_path=checkpoint_load_path,
            tboard_logs_load_path=tboard_logs_load_path,
            num_loop_iterations=num_loop_iterations,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Walk Training")
    parser.add_argument("--dataset", type=str, nargs="?", default=None)
    parser.add_argument(
        "--additional_layer",
        type=str,
        choices=["identity", "hyperbolic", "euclidean"],
        default="identity",
    )
    parser.add_argument("--learning_rate", type=float, default=0.3)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--knit5_checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_save_path", type=str, default=None)
    parser.add_argument("--tboard_logs_save_path", type=str, default=None)
    parser.add_argument("--checkpoint_load_path", type=str, default=None)
    parser.add_argument("--tboard_logs_load_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--additional_layer_lr", type=float, default=0.001)
    parser.add_argument(
        "--no_soft_prompt", action="store_true", default=False
    )
    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_loop_iterations", type=int, default=1)
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
    num_layers = args.num_layers
    num_loop_iterations = args.num_loop_iterations
    checkpoint_load_path = args.checkpoint_load_path
    tboard_logs_load_path = args.tboard_logs_load_path

    gpu_parallel = config.random_walk_training.gpu_parallelization
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
                no_soft_prompt,
                use_scheduler,
                num_layers,
                checkpoint_load_path,
                tboard_logs_load_path,
                num_loop_iterations,
            ),
            nprocs=world_size,
            join=True,
        )
    else:
        _train_random_walk(
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
            no_soft_prompt=no_soft_prompt,
            use_scheduler=use_scheduler,
            num_layers=num_layers,
            checkpoint_load_path=checkpoint_load_path,
            tboard_logs_load_path=tboard_logs_load_path,
            num_loop_iterations=num_loop_iterations,
        )
