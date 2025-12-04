from transformers import AutoTokenizer
import torch
from src.config import Config
from torch.utils.data import DataLoader


from src.models import SoftPromptModel, T5ModelWithAdditionalLayer
from src.eval import  evaluate_random_walk_training
import argparse
from src.datasets import get_random_walk_dataset
import pandas as pd
def test_random_walk(dataset, additional_layer, batch_size, knit5_checkpoint_path, prompt_tuning_checkpoint_path):
    GPU_PARALLELIZATION = True# if dataset in ['2wikimultihop', 'wikimultihop', '2wikihop', 'wikihop'] else True
    WITH_MODEL_STATE_DICT = GPU_PARALLELIZATION
    _, _, random_walk_test = get_random_walk_dataset(dataset)

    #Specify Hyperparameters via config file
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')

    test_random_walk_dataloader = DataLoader(random_walk_test, batch_size=batch_size, shuffle=False)


    #google/t5-large-lm-adapt
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    hyperbolic_knit5_model = T5ModelWithAdditionalLayer(num_layers=1, layer_type=additional_layer, curvature=config.random_walk_training.curvature, checkpoint_hyperbolic_knit5=knit5_checkpoint_path, with_model_state_dict=WITH_MODEL_STATE_DICT, gpu_parallelization=GPU_PARALLELIZATION, soft_prompt_length=config.random_walk_training.prompt_length)
    import torch.nn as nn

    checkpoint = torch.load(prompt_tuning_checkpoint_path, map_location=device)
    hopping_prompt = nn.Parameter(checkpoint['soft_prompt_state_dict'], requires_grad=False)
    additional_linear_layer = checkpoint['additional_linear_layer']

    hyperbolic_knit5_model.hyperbolic_layer.load_state_dict(additional_linear_layer)
    print("Loaded Soft Prompts and Additional Linear Layer")
    print(f"Loaded Soft Prompt and Additional Layer from: {prompt_tuning_checkpoint_path}")

    print(f"{hopping_prompt.shape = }")
    model = SoftPromptModel(knit5=hyperbolic_knit5_model, soft_prompt=hopping_prompt)




    pred_vs_label = evaluate_random_walk_training(model, tokenizer, test_random_walk_dataloader)
    
    print(pred_vs_label)
    df = pd.DataFrame(pred_vs_label)
    df.to_csv('pred_vs_label.csv', sep=';')


  


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
        '--knit5_checkpoint_path',
        type=str,
        default=None,
        help='Specify Checkpoint Path of finetuned KNIT5 Model'
    )
    parser.add_argument(
        '--prompt_tuning_checkpoint_path',
        type=str,
        default=None,
        help='Specify checkpoint path of prompt tuning experiments'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Specify batch size'
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    dataset = args.dataset 
    additional_layer = args.additional_layer
    knit5_checkpoint_path = args.knit5_checkpoint_path
    prompt_tuning_checkpoint_path = args.prompt_tuning_checkpoint_path
    batch_size = args.batch_size
    test_random_walk(dataset, additional_layer, batch_size, knit5_checkpoint_path, prompt_tuning_checkpoint_path)