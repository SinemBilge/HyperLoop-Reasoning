from src.utils.util import load_dataset, load_c4_dataset
from src.knowledge_graph import create_knowledge_graph_metaqa, create_knowledge_graph_mlpq, create_knowledge_graph_pql

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import KnowledgeIntegrationMLPQDataset, KnowledgeIntegrationMetaQADataset, KnowledgeIntegrationPQLDataset, KnowledgeIntegrationWikiHopDataset, C4Dataset
import torch
from src.config import Config
from torch.utils.data import DataLoader
from src.train import *
import argparse
from src.models import T5ModelWithAdditionalLayer
import torch.distributed as dist
from math import log,exp

config = Config()
def _knowledge_integration_with_c4(dataset, rank, world_size, learning_rate=0.001, epochs=50, checkpoint_save_path=None, tboard_logs_save_path=None, batch_size=64, additional_layer = 'identity', curvature = 1.0):

    if dataset in ['2wikimultihop', 'wikimultihop', '2wikihop', 'wikihop']: 
        print("Training on 2WikiMultiHopQA")
        train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset("dataset/2wikimultihop", do_correct_wrong_evidences=True)


        print("Creating Single Hop Datasets...")
        dataset_with_all_entries = pd.concat([train_dataset, dev_dataset, test_dataset])

        ki_dataset = KnowledgeIntegrationWikiHopDataset(dataset_with_all_entries)

    elif dataset in ['metaqa']:
        print("Training on MetaQA")
        #metaqa_kg_data = pd.read_csv('dataset/metaqa/kb.txt', sep="|")

        #Comment Out if we want to use the whole kb. This used only the kb for questions with a single answer
        df_dev = pd.read_json("dataset/metaqa/2hops/qa_dev_evidences.json")
        df_train = pd.read_json("dataset/metaqa/2hops/qa_train_evidences.json")
        df_test = pd.read_json("dataset/metaqa/2hops/qa_test_evidences.json")
        max_answers = 1
        df_kg = pd.concat([df_dev, df_train, df_test])
        kg = create_knowledge_graph_metaqa(df_kg, from_kb=False, max_answers=max_answers)

        ki_dataset = KnowledgeIntegrationMetaQADataset(kg)
    elif dataset in ['mlpq']:
        #txt_file_paths = ['dataset/mlpq/Triples_in_questions/EN_KG', 'dataset/mlpq/Triples_in_questions/FR_KG']
        train_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_train_question_evidences.json', lines=True)
        validation_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_dev_question_evidences.json', lines=True)
        test_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_test_question_evidences.json', lines=True)

        df_kg = pd.concat([train_dataframe, validation_dataframe, test_dataframe])
        kg = create_knowledge_graph_mlpq(df_kg, from_kb = False, hops=2)
        ki_dataset = KnowledgeIntegrationMLPQDataset(kg)
    elif dataset in ['mlpq-3hop']:
        train_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/3-hop/3hop_train_question_evidences.json', lines=True)
        validation_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/3-hop/3hop_dev_question_evidences.json', lines=True)
        test_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/3-hop/3hop_test_question_evidences.json', lines=True)

        df_kg = pd.concat([train_dataframe, validation_dataframe, test_dataframe])
        kg = create_knowledge_graph_mlpq(df_kg, from_kb = False, hops=3)
        ki_dataset = KnowledgeIntegrationMLPQDataset(kg)
    elif dataset in ['pql']:
        # from src.utils.util import load_train_test_pql_dataset
        file_path_kb = "dataset/pathquestion/PQL-KB.txt"
        file_path_paths = "dataset/pathquestion/PQ-2H.txt"
        kg = create_knowledge_graph_pql(file_path_paths, from_kb=False)
        ki_dataset = KnowledgeIntegrationPQLDataset(kg)
    elif dataset in ['pql-3hop']:
        # from src.utils.util import load_train_test_pql_dataset
        # file_path_kb = "dataset/pathquestion/PQL-KB.txt"
        file_path_paths = "dataset/pathquestion/PQ-3H.txt"
        kg = create_knowledge_graph_pql(file_path_paths, from_kb=False)
        ki_dataset = KnowledgeIntegrationPQLDataset(kg)
    else:
        raise ValueError("Unknown Dataset")  


    ki_train = ki_dataset

    #Specify Hyperparameters via config file
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    print(f"Process rank: {rank} using device: {device}")
    if dist.is_initialized():
        print(f"Rank: {dist.get_rank()} / World size: {dist.get_world_size()}")

    #Define Tokenizer and Model
    #google/t5-large-lm-adapt
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.t5_model.model_name)
    config.t5_model.tokenizer_max_length = 128
    tokenizer.model_max_length = 128#config.t5_model.tokenizer_max_length
    print(f"Loading Model {config.t5_model.model_name}...")
    #Adjust Dropout
    config.single_hop_training.curvature = log(exp(curvature) - 1)
    print(f"Setting Curvature to {config.single_hop_training.curvature}")

    print(f"Train Euclidean T5 Model")
    model = T5ModelWithAdditionalLayer(layer_type=additional_layer, curvature=config.single_hop_training.curvature, checkpoint_hyperbolic_knit5=config.single_hop_training.model_checkpoint_path, with_model_state_dict=True, gpu_parallelization=config.single_hop_training.gpu_parallelization)
    model.config.dropout_rate = 0.1
    model.config.hidden_dropout_prob = 0.1
    model.config.attention_probs_dropout_prob = 0.1
    model.config.classifier_dropout = 0.1

    base_path = 'c4/en/c4-train.{:05d}-of-01024.json'
    c4_dataset = load_c4_dataset(base_path, number_of_files=3)


    c4_dataset = c4_dataset[:len(ki_train)]

    objective = 'prefix_language_modeling'
    c4_train = C4Dataset(c4_dataset ,tokenizer=tokenizer, objective=objective)
    
    #Set Variables
    config.single_hop_training.learning_rate = learning_rate
    print(f"Setting Learning Rate to {learning_rate}")
    config.single_hop_training.epochs = epochs
    print(f"Setting Number of Epochs to {epochs}")
    config.single_hop_training.model_save_path = checkpoint_save_path
    print(f"Setting Checkpoint Save Path to {checkpoint_save_path}")
    config.single_hop_training.log_dir = tboard_logs_save_path
    print(f"Setting tboard_logs_save_path to {tboard_logs_save_path}")
    config.t5_model.batch_size = batch_size
    print(f"Setting Batch Size to {batch_size}")


    from torch.utils.data import ConcatDataset
    

    if config.single_hop_training.gpu_parallelization:
        from torch.utils.data import DistributedSampler

        # Create DistributedSampler for each dataset
        ki_train_sampler = DistributedSampler(ki_train, shuffle=True, num_replicas=world_size, rank=rank)
        c4_sampler = DistributedSampler(c4_train,  shuffle=True, num_replicas=world_size, rank=rank)
        ki_dev_sampler = DistributedSampler(ki_train, shuffle=False, num_replicas=world_size, rank=rank)

        # Create DataLoaders using these samplers
        single_hop_dataloader_train = DataLoader(ki_train, sampler=ki_train_sampler, batch_size=config.t5_model.batch_size//world_size)
        c4_dataloader_train = DataLoader(c4_train, sampler=c4_sampler, batch_size=config.t5_model.batch_size//world_size)
        single_hop_dataloader_dev = DataLoader(ki_train, sampler=ki_dev_sampler, batch_size=config.t5_model.batch_size//world_size)
    else:
        c4_dataloader_train = DataLoader(c4_train, batch_size=config.t5_model.batch_size, shuffle=True, num_workers=config.single_hop_training.num_workers)
        single_hop_dataloader_train = DataLoader(ki_train, batch_size=config.t5_model.batch_size, shuffle=True, num_workers=config.single_hop_training.num_workers)
        single_hop_dataloader_dev = DataLoader(ki_train,  batch_size=config.t5_model.batch_size, shuffle=False, num_workers=config.single_hop_training.num_workers)


    config.single_hop_training.additional_log_info = config.single_hop_training.additional_log_info + f"_{additional_layer}_c{curvature}" 
    trainer = ModelTrainer(model,
                        tokenizer,
                        [single_hop_dataloader_train, c4_dataloader_train],
                        single_hop_dataloader_dev,
                        config,
                        device=device,
                        validation_step=1,
                        checkpoint_path=config.single_hop_training.model_checkpoint_path,
                        tboard_checkpoint_path=config.single_hop_training.tboard_checkpoint_path,
                        method = 'single_hop_training',
                        gpu_parallelization=config.single_hop_training.gpu_parallelization,
                        rank = rank)

    print(f'Knowledge Integration training..')
    print(f'with model: {config.t5_model.model_name}')
    print(f'Model Config: {model.config}')
    print(f'for: {config.single_hop_training.epochs} epochs')
    print(f'with batch size: {config.t5_model.batch_size}')
    print(f'with optimizer: {config.single_hop_training.optimizer}')
    print(f'With C4')

    print(f'Training with {len(ki_train)} Triples for Knowledge Integration.')
    trainer.train(epochs=config.single_hop_training.epochs)


import torch.distributed as dist
import torch.multiprocessing as mp
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
    

def train_ddp(rank, world_size, dataset, learning_rate, epochs, checkpoint_save_path, tboard_logs_save_path, batch_size, additional_layer, curvature):
    setup_ddp(rank, world_size)
    _knowledge_integration_with_c4(dataset=dataset, rank=rank, world_size=world_size, learning_rate=learning_rate, epochs=epochs, checkpoint_save_path=checkpoint_save_path, tboard_logs_save_path=tboard_logs_save_path, batch_size=batch_size, additional_layer = additional_layer, curvature = curvature)  # Call your training method
    dist.destroy_process_group()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge Integration Training')
    parser.add_argument('--dataset', type=str, default=None, help='Specify the Dataset (e.g., metaqa, 2wikimultihop)')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.3,  # You can set a different default if needed
        help='Specify the Learning Rate'
    )  
    parser.add_argument(
        '--curvature',
        type=float,
        default=1.0,  # You can set a different default if needed
        help='Specify the Curvature'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Specify number of epochs'
    )
    parser.add_argument(
        '--checkpoint_save_path',
        type=str,
        default=None,
        help='Specify save path for the checkpoint'
    )
    parser.add_argument(
        '--additional_layer',
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
        '--batch_size',
        type=int,
        default=64,
        help='Specify path for tensorboard logs'
    )
    args = parser.parse_args()
    dataset = args.dataset  # Pass the dataset name
    learning_rate = args.learning_rate
    epochs = args.epochs
    checkpoint_save_path = args.checkpoint_save_path
    tboard_logs_save_path = args.tboard_logs_save_path
    batch_size = args.batch_size
    additional_layer = args.additional_layer
    curvature = args.curvature
    if config.single_hop_training.gpu_parallelization:
        world_size = torch.cuda.device_count()
        mp.spawn(train_ddp, args=(world_size, dataset, learning_rate, epochs, checkpoint_save_path, tboard_logs_save_path, batch_size, additional_layer, curvature), nprocs=world_size, join=True)
    else:
        _knowledge_integration_with_c4(dataset=dataset, rank=0, world_size=1, learning_rate=learning_rate, epochs=epochs, checkpoint_save_path=checkpoint_save_path, tboard_logs_save_path=tboard_logs_save_path, batch_size=batch_size, additional_layer=additional_layer, curvature=curvature)
