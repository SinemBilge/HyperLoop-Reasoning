from src.utils.util import load_dataset, get_top_token_embeddings, load_train_test_pql_dataset
import pandas as pd
from src.train.soft_prompt_trainer import SoftPromptTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from src.config import Config
from torch.utils.data import DataLoader
from src.knowledge_graph import create_knowledge_graph_wikimultihop, create_knowledge_graph_metaqa, create_knowledge_graph_mlpq, create_knowledge_graph_pql
from src.models import SoftPromptModel, T5ModelWithAdditionalLayer
from src.datasets import RandomWalkMetaQADataset, RandomWalkMLPQDataset, RandomWalkWikiHopDataset, RandomWalkPQLDataset
import argparse
import os
from math import exp, log
from src.datasets.dataloader import get_random_walk_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"
config = Config()
def _train_random_walk(additional_layer : str, dataset : str, rank, world_size, lr = 0.3, curvature = 1.0, knit5_checkpoint_path=None, checkpoint_save_path = None, tboard_logs_save_path = None, epochs = None, batch_size = 128, additional_layer_lr = 0.001, no_soft_prompt = False, use_scheduler = False, num_layers = 1, checkpoint_load_path = None, tboard_logs_load_path = None):
    GPU_PARALLELIZATION = True #if dataset in ['2wikimultihop', 'wikimultihop', '2wikihop', 'wikihop'] else True
    WITH_MODEL_STATE_DICT = GPU_PARALLELIZATION
    random_walk_train, random_walk_dev, _ = get_random_walk_dataset(dataset)
    print(f"Number of Random Walks Train: {len(random_walk_train)}")
    print(f"Number of Random Walk Dev: {len(random_walk_dev)}")
    #Specify Hyperparameters via config file
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
        random_walk_dataloader_dev = DataLoader(random_walk_dev,  sampler=random_walk_sampler_dev, batch_size=config.t5_model.batch_size//world_size, num_workers=config.random_walk_training.num_workers)
    else:
        random_walk_dataloader_train = DataLoader(random_walk_train, batch_size=config.t5_model.batch_size, shuffle=True, num_workers=config.random_walk_training.num_workers)
        random_walk_dataloader_dev = DataLoader(random_walk_dev, batch_size=config.t5_model.batch_size, shuffle=False, num_workers=config.random_walk_training.num_workers)
    #google/t5-large-lm-adapt
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    longest_sequence = 0
        # # Iterate through random_walk_dev dataset
    for incomplete, complete in random_walk_dev:
        # Tokenize the sample text
        tokens = tokenizer(complete, return_tensors='pt', truncation=False).input_ids
        token_length = tokens.size(1)  # Get the token length
        # Update the longest token sequence for dev
        if token_length > longest_sequence:
            longest_sequence = token_length
    for incomplete, complete in random_walk_train:
        # Tokenize the sample text
        tokens = tokenizer(complete, return_tensors='pt', truncation=False).input_ids
        token_length = tokens.size(1)  # Get the token length
        # Update the longest token sequence for dev
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

    print("Loading Model...")
    config.random_walk_training.model_checkpoint_path = knit5_checkpoint_path
    print(f"Setting KNIT5 Checkpoint Load Path to: {knit5_checkpoint_path}")
    print(f"Number of Layers are {num_layers}")
    hyperbolic_knit5_model = T5ModelWithAdditionalLayer(layer_type=additional_layer, num_layers=num_layers, curvature=config.random_walk_training.curvature, checkpoint_hyperbolic_knit5=config.random_walk_training.model_checkpoint_path, with_model_state_dict=WITH_MODEL_STATE_DICT, gpu_parallelization=GPU_PARALLELIZATION, soft_prompt_length=config.random_walk_training.prompt_length)
    model = SoftPromptModel(knit5=hyperbolic_knit5_model, model_name='hyperbolic_hopping_prompt', soft_prompt_length=config.random_walk_training.prompt_length)
    print(f"Train with hyperbolic Soft Prompt Model with additional layer {additional_layer} and curvature {config.random_walk_training.curvature if additional_layer == 'hyperbolic' else 0.0}")
    # Count trainable parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {num_trainable_params}")
    if epochs is not None:
        config.random_walk_training.epochs = epochs
        print(f"Setting epochs to: {epochs}")
    print("Model type before passing to SoftPromptTrainer:", type(model))
    if checkpoint_save_path is not None:
        config.random_walk_training.model_save_path = checkpoint_save_path
        print(f"Setting Checkpoint Save path to: {checkpoint_save_path}")
    if tboard_logs_save_path is not None:
        config.random_walk_training.log_dir = tboard_logs_save_path
        print(f"Setting Tensorboard Log save path to: {tboard_logs_save_path}")

    config.random_walk_training.additional_log_info=f'{additional_layer}_after_encoder_bsize{config.t5_model.batch_size}_prompt_lenght{config.random_walk_training.prompt_length}_lr{config.random_walk_training.learning_rate}_curvature{model.knit5.hyperbolic_layer.manifold.c.item() if additional_layer == "hyperbolic" else 0}_additional_layer_lr{additional_layer_lr}_use_prompt_{config.random_walk_training.use_soft_prompt}{"_cont" if config.random_walk_training.hopping_prompt_checkpoint_path else ""}'
    trainer = SoftPromptTrainer(model,
                      tokenizer,
                      random_walk_dataloader_train,
                      random_walk_dataloader_dev,
                      config,
                      device=device,
                      method='random_walk_training',
                      checkpoint_path=config.random_walk_training.hopping_prompt_checkpoint_path,
                      tboard_checkpoint_path=config.random_walk_training.tboard_checkpoint_path,
                      retrain = True,
                      gpu_parallelization=config.random_walk_training.gpu_parallelization,
                      rank = rank
                      )


    print(f'Random Walk Training..')
    print(f'with model: {config.t5_model.model_name}')
    print(f'with lr: {config.random_walk_training.learning_rate}')

    # print(f'Model Config: {model.knit5.config}')
    print(f'for: {config.random_walk_training.epochs} epochs')
    print(f'with effective batch size: {config.t5_model.batch_size} ({config.t5_model.batch_size / world_size} per GPU)')
    print(f'with optimizer: {config.random_walk_training.optimizer}')

    trainer.train(epochs=config.random_walk_training.epochs)  

import torch.distributed as dist
import torch.multiprocessing as mp
import os
from src.utils.util import set_seed
def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)
    set_seed(42)

def train_ddp(rank, world_size, dataset, additional_layer, lr, curvature, knit5_checkpoint_path, checkpoint_save_path, tboard_logs_save_path, epochs, batch_size, additional_layer_lr, no_soft_prompt, use_scheduler, num_layers, checkpoint_load_path, tboard_logs_load_path):
    setup_ddp(rank, world_size)
    _train_random_walk(additional_layer=additional_layer, dataset=dataset, rank=rank, world_size=world_size, lr=lr, curvature=curvature, knit5_checkpoint_path=knit5_checkpoint_path, checkpoint_save_path=checkpoint_save_path, tboard_logs_save_path=tboard_logs_save_path, epochs = epochs, batch_size = batch_size, additional_layer_lr = additional_layer_lr, no_soft_prompt = no_soft_prompt, use_scheduler = use_scheduler, num_layers = num_layers, checkpoint_load_path = checkpoint_load_path, tboard_logs_load_path = tboard_logs_load_path)  # Call your training method
    dist.destroy_process_group()

if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser(description='Random Walk Training Training')
    parser.add_argument('--dataset', type=str, nargs='?', default=None, help='Specify the dataset (e.g., metaqa, 2wikimultihop)')
    # New argument: --additional_layer
    parser.add_argument(
        '--additional_layer',
        type=str,
        choices=['identity', 'hyperbolic', 'linear'],
        default='identity',  # You can set a different default if needed
        help='Specify the type of additional layer to use: identity, hyperbolic, or linear'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.3,  # You can set a different default if needed
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
        help='Specify path for tensorboard logs'
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
        default = False
    )
    parser.add_argument(
        '--use_scheduler',
        action='store_true',
        help='If set, use scheduler'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=1,
        help='Specify number of layers'
    )
    

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    dataset = args.dataset  # Pass the dataset name
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
    checkpoint_load_path = args.checkpoint_load_path
    tboard_logs_load_path = args.tboard_logs_load_path
    mp.spawn(train_ddp, args=(world_size, dataset, additional_layer, lr, curvature, knit5_checkpoint_path, checkpoint_save_path, tboard_logs_save_path, epochs, batch_size, additional_layer_lr, no_soft_prompt, use_scheduler, num_layers, checkpoint_load_path, tboard_logs_load_path), nprocs=world_size, join=True)

