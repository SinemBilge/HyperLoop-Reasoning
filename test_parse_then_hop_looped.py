"""
Evaluation script for Parse-Then-Hop with Looped Hyperbolic Transformer.

This script tests the full end-to-end pipeline:
    Question -> Parsing Model -> Incomplete Path -> Hopping Model -> Complete Path

Usage:
    python test_parse_then_hop_looped.py \
        --dataset metaqa \
        --additional_layer_parse hyperbolic \
        --additional_layer_hop hyperbolic \
        --knit5_checkpoint_path checkpoints/knowledge_integration/knit5.pth \
        --parsing_prompt_checkpoint_path checkpoints/parse/soft_prompt.pth \
        --hopping_prompt_checkpoint_path checkpoints/hop/soft_prompt.pth \
        --num_loops 4 \
        --batch_size 8

"""

from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import pandas as pd
import json

from src.config import Config
from src.models import SoftPromptModel
from src.models.hyperbolic_t5_with_looping import T5ModelWithLoopedHyperbolic
from src.eval import evaluate_parse_then_hop_training
from src.datasets import get_parse_then_hop_test_dataset


def test_parse_then_hop_looped(
    dataset,
    additional_layer_parse,
    additional_layer_hop,
    batch_size,
    knit5_checkpoint_path,
    hopping_prompt_checkpoint_path,
    parsing_prompt_checkpoint_path,
    num_loops=4,
    max_norm=0.9,
    input_injection=True
):
    """
    Test the full parse-then-hop pipeline using looped hyperbolic T5 models.
    """
    GPU_PARALLELIZATION = True
    WITH_MODEL_STATE_DICT = GPU_PARALLELIZATION

    # Load test dataset
    parse_then_hop_test = get_parse_then_hop_test_dataset(dataset)

    # Load config
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Evaluating on device: {device}')

    path_test = DataLoader(parse_then_hop_test, batch_size=batch_size, shuffle=False, num_workers=1)

    # Load tokenizer
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize models with looping
    print("Loading Parsing Model WITH LOOPING...")
    print(f"  Layer type: {additional_layer_parse}")
    print(f"  Num loops: {num_loops}, Max norm: {max_norm}, Input injection: {input_injection}")

    parsing_knit5_model = T5ModelWithLoopedHyperbolic(
        layer_type=additional_layer_parse,
        curvature=config.random_walk_training.curvature,
        checkpoint_hyperbolic_knit5=knit5_checkpoint_path,
        with_model_state_dict=WITH_MODEL_STATE_DICT,
        gpu_parallelization=GPU_PARALLELIZATION,
        soft_prompt_length=config.random_walk_training.prompt_length,
        num_loops=num_loops,
        max_norm=max_norm,
        input_injection=input_injection
    )

    print("Loading Hopping Model WITH LOOPING...")
    print(f"  Layer type: {additional_layer_hop}")
    print(f"  Num loops: {num_loops}, Max norm: {max_norm}, Input injection: {input_injection}")

    hopping_knit5_model = T5ModelWithLoopedHyperbolic(
        layer_type=additional_layer_hop,
        curvature=config.random_walk_training.curvature,
        checkpoint_hyperbolic_knit5=knit5_checkpoint_path,
        with_model_state_dict=WITH_MODEL_STATE_DICT,
        gpu_parallelization=GPU_PARALLELIZATION,
        soft_prompt_length=config.random_walk_training.prompt_length,
        num_loops=num_loops,
        max_norm=max_norm,
        input_injection=input_injection
    )

    # Load parsing checkpoint
    print(f"\nLoading parsing checkpoint from: {parsing_prompt_checkpoint_path}")
    checkpoint = torch.load(parsing_prompt_checkpoint_path, map_location=device)
    parsing_prompt = nn.Parameter(checkpoint['soft_prompt_state_dict'], requires_grad=False)
    additional_parsing_linear_layer = checkpoint['additional_linear_layer']

    # Load hopping checkpoint
    print(f"Loading hopping checkpoint from: {hopping_prompt_checkpoint_path}")
    checkpoint = torch.load(hopping_prompt_checkpoint_path, map_location=device)
    hopping_prompt = nn.Parameter(checkpoint['soft_prompt_state_dict'], requires_grad=False)
    additional_hopping_linear_layer = checkpoint['additional_linear_layer']

    # Load hyperbolic layer weights
    if additional_layer_hop != 'identity':
        hopping_knit5_model.hyperbolic_layer.load_state_dict(additional_hopping_linear_layer)
    if additional_layer_parse != 'identity':
        parsing_knit5_model.hyperbolic_layer.load_state_dict(additional_parsing_linear_layer)

    print("Loaded Soft Prompts and Additional Linear Layers")
    print(f"Loaded Parsing Soft Prompt from {parsing_prompt_checkpoint_path}")
    print(f"Loaded Hopping Soft Prompt from {hopping_prompt_checkpoint_path}")

    print(f"{parsing_prompt.shape = }")
    print(f"{hopping_prompt.shape = }")

    # Create SoftPromptModel wrappers
    parsing_model = SoftPromptModel(knit5=parsing_knit5_model, soft_prompt=parsing_prompt)
    hopping_model = SoftPromptModel(knit5=hopping_knit5_model, soft_prompt=hopping_prompt)

    # Run evaluation
    print("\n" + "="*60)
    print("Starting Parse-Then-Hop Evaluation WITH LOOPING")
    print("="*60)

    pred_vs_label = evaluate_parse_then_hop_training(
        parsing_model=parsing_model,
        hopping_model=hopping_model,
        tokenizer=tokenizer,
        test_dataloader=path_test,
        do_extract_answer=False
    )

    # Save predictions
    df = pd.DataFrame(pred_vs_label)
    df.to_csv('pred_vs_label_path.csv', sep=';')
    print(f"\nPredictions saved to pred_vs_label_path.csv")

    return pred_vs_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse-Then-Hop Evaluation with Looping')

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Specify the dataset (e.g., metaqa, 2wikimultihop)'
    )

    parser.add_argument(
        '--additional_layer_parse',
        type=str,
        choices=['identity', 'hyperbolic', 'linear'],
        default='hyperbolic',
        help='Type of additional layer for parsing model'
    )

    parser.add_argument(
        '--additional_layer_hop',
        type=str,
        choices=['identity', 'hyperbolic', 'linear'],
        default='hyperbolic',
        help='Type of additional layer for hopping model'
    )

    parser.add_argument(
        '--knit5_checkpoint_path',
        type=str,
        default=None,
        help='Checkpoint path of finetuned KNIT5 Model'
    )

    parser.add_argument(
        '--parsing_prompt_checkpoint_path',
        type=str,
        required=True,
        help='Checkpoint path for parsing soft prompt'
    )

    parser.add_argument(
        '--hopping_prompt_checkpoint_path',
        type=str,
        required=True,
        help='Checkpoint path for hopping soft prompt'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for evaluation'
    )

    # Looping parameters
    parser.add_argument(
        '--num_loops',
        type=int,
        default=4,
        help='Number of times to loop the hyperbolic layer (default: 4)'
    )

    parser.add_argument(
        '--max_norm',
        type=float,
        default=0.9,
        help='Max norm for Poincare ball clamping (default: 0.9)'
    )

    parser.add_argument(
        '--no_input_injection',
        action='store_true',
        help='Disable input injection in looping'
    )

    args = parser.parse_args()

    input_injection = not args.no_input_injection

    test_parse_then_hop_looped(
        dataset=args.dataset,
        additional_layer_parse=args.additional_layer_parse,
        additional_layer_hop=args.additional_layer_hop,
        batch_size=args.batch_size,
        knit5_checkpoint_path=args.knit5_checkpoint_path,
        hopping_prompt_checkpoint_path=args.hopping_prompt_checkpoint_path,
        parsing_prompt_checkpoint_path=args.parsing_prompt_checkpoint_path,
        num_loops=args.num_loops,
        max_norm=args.max_norm,
        input_injection=input_injection
    )
