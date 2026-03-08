from transformers import AutoTokenizer
import torch
from src.config import Config
from torch.utils.data import DataLoader
from src.models import SoftPromptModel, T5ModelWithAdditionalLayer
from src.eval import evaluate_parse_then_hop_training
import argparse
from src.datasets import get_parse_then_hop_test_dataset
import pandas as pd

def test_path(dataset, additional_layer_parse, additional_layer_hop, batch_size,
              knit5_checkpoint_path, hopping_prompt_checkpoint_path,
              parsing_prompt_checkpoint_path,
              num_loop_iterations_parse=1, num_loop_iterations_hop=1):
    parse_then_hop_test = get_parse_then_hop_test_dataset(dataset)

    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')

    path_test = DataLoader(parse_then_hop_test, batch_size=batch_size, shuffle=False, num_workers=1)

    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    parsring_knit5_model = T5ModelWithAdditionalLayer(
        layer_type=additional_layer_parse,
        curvature=config.parse_then_hop_training.curvature,
        checkpoint_hyperbolic_knit5=knit5_checkpoint_path,
        num_loop_iterations=num_loop_iterations_parse,
    )
    hopping_knit5_model = T5ModelWithAdditionalLayer(
        num_layers=1,
        layer_type=additional_layer_hop,
        curvature=config.random_walk_training.curvature,
        checkpoint_hyperbolic_knit5=knit5_checkpoint_path,
        num_loop_iterations=num_loop_iterations_hop,
    )
    import torch.nn as nn

    checkpoint = torch.load(parsing_prompt_checkpoint_path, map_location=device)
    parsing_prompt = nn.Parameter(checkpoint['soft_prompt_state_dict'], requires_grad=False)
    additional_parsing_linear_layer = checkpoint['additional_linear_layer']

    checkpoint = torch.load(hopping_prompt_checkpoint_path, map_location=device)
    hopping_prompt = nn.Parameter(checkpoint['soft_prompt_state_dict'], requires_grad=False)
    additional_hopping_linear_layer = checkpoint['additional_linear_layer']

    parsring_knit5_model.additional_layer.load_state_dict(additional_parsing_linear_layer, strict=False)
    hopping_knit5_model.additional_layer.load_state_dict(additional_hopping_linear_layer, strict=False)
    print("Loaded Soft Prompts and Additional Linear Layer")
    print(f"Loaded Parsing Soft Prompt from {parsing_prompt_checkpoint_path}")
    print(f"Loaded Hopping Soft Prompt from {hopping_prompt_checkpoint_path}")

    print(f"{parsing_prompt.shape = }")
    print(f"{hopping_prompt.shape = }")
    parsing_model = SoftPromptModel(knit5=parsring_knit5_model, soft_prompt=parsing_prompt)
    hopping_model = SoftPromptModel(knit5=hopping_knit5_model, soft_prompt=hopping_prompt)

    pred_vs_label = evaluate_parse_then_hop_training(parsing_model=parsing_model,
                                     hopping_model=hopping_model,
                                     tokenizer=tokenizer,
                                     test_dataloader=path_test,
                                     do_extract_answer=False)
    print(pred_vs_label)
    df = pd.DataFrame(pred_vs_label)
    df.to_csv('pred_vs_label_path.csv', sep=';')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse Then Hop Testing')
    parser.add_argument('--dataset', type=str, nargs='?', default=None, help='Specify the dataset (e.g., metaqa, 2wikimultihop)')
    parser.add_argument(
        '--additional_layer_parse',
        type=str,
        choices=['identity', 'hyperbolic', 'euclidean'],
        default='identity',
        help='Specify the type of additional layer for parsing'
    )
    parser.add_argument(
        '--additional_layer_hop',
        type=str,
        choices=['identity', 'hyperbolic', 'euclidean'],
        default='identity',
        help='Specify the type of additional layer for hopping'
    )
    parser.add_argument(
        '--knit5_checkpoint_path',
        type=str,
        default=None,
        help='Specify Checkpoint Path of finetuned KNIT5 Model'
    )
    parser.add_argument(
        '--parsing_prompt_checkpoint_path',
        type=str,
        default=None,
        help='Specify checkpoint path of parsing prompt'
    )
    parser.add_argument(
        '--hopping_prompt_checkpoint_path',
        type=str,
        default=None,
        help='Specify checkpoint path of hopping prompt'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Specify batch size'
    )
    parser.add_argument(
        '--num_loop_iterations_parse',
        type=int,
        default=1,
        help='Number of loop iterations for parsing model (1=original, 2+=looped)'
    )
    parser.add_argument(
        '--num_loop_iterations_hop',
        type=int,
        default=1,
        help='Number of loop iterations for hopping model (1=original, 2+=looped)'
    )
    args = parser.parse_args()

    test_path(
        dataset=args.dataset,
        additional_layer_parse=args.additional_layer_parse,
        additional_layer_hop=args.additional_layer_hop,
        batch_size=args.batch_size,
        knit5_checkpoint_path=args.knit5_checkpoint_path,
        hopping_prompt_checkpoint_path=args.hopping_prompt_checkpoint_path,
        parsing_prompt_checkpoint_path=args.parsing_prompt_checkpoint_path,
        num_loop_iterations_parse=args.num_loop_iterations_parse,
        num_loop_iterations_hop=args.num_loop_iterations_hop,
    )
