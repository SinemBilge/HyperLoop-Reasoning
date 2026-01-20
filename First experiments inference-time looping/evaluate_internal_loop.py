"""
Evaluate Internal Looped Hyperbolic Model

This uses the LoopedHyperbolicT5 which loops the hyperbolic layer transformation
internally (on encoder hidden states), NOT at the generation level.

Flow:
  Question -> Encoder -> [Hyperbolic Layer x N] -> Decoder -> Answer
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn as nn
import argparse
from tqdm import tqdm
import pandas as pd
import json
import re
import string
from collections import Counter

from src.models import LoopedHyperbolicT5, SoftPromptModel
from src.datasets import get_parse_then_hop_test_dataset
from src.config import Config


def normalize_answer(s):
    """Normalize answer for comparison (from 2WikiMultiHop eval)"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0, 0, 0
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0, 0, 0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def extract_answer(path):
    """Extract final answer from path: 'e1 ; r1 ; e2 ; r2 ; answer'"""
    parts = path.split(';')
    return parts[-1].strip() if parts else path


def create_looped_model(
    knit5_checkpoint_path: str,
    prompt_checkpoint_path: str,
    num_loops: int = 1,
    use_residual: bool = True,
    device: str = 'cuda'
):
    """Create a looped hyperbolic model with internal looping"""
    config = Config()

    # Create model with internal looping
    model = LoopedHyperbolicT5(
        layer_type='hyperbolic',
        curvature=config.random_walk_training.curvature,
        checkpoint_hyperbolic_knit5=knit5_checkpoint_path,
        with_model_state_dict=True,
        gpu_parallelization=True,
        soft_prompt_length=config.random_walk_training.prompt_length,
        num_loops=num_loops,
        use_residual=use_residual
    )

    # Load soft prompt and additional layer from checkpoint
    checkpoint = torch.load(prompt_checkpoint_path, map_location=device)
    soft_prompt = nn.Parameter(checkpoint['soft_prompt_state_dict'], requires_grad=False)
    model.hyperbolic_layer.load_state_dict(checkpoint['additional_linear_layer'])

    # Wrap with soft prompt
    soft_prompt_model = SoftPromptModel(knit5=model, soft_prompt=soft_prompt)

    return soft_prompt_model


def evaluate(
    dataset_name: str,
    knit5_checkpoint_path: str,
    hopping_prompt_checkpoint_path: str,
    num_loops: int = 1,
    use_residual: bool = True,
    batch_size: int = 8,
    output_file: str = 'internal_loop_results.csv'
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained('google/t5-large-lm-adapt')

    print(f"\nCreating model with {num_loops} internal loop(s), residual={use_residual}")
    model = create_looped_model(
        knit5_checkpoint_path=knit5_checkpoint_path,
        prompt_checkpoint_path=hopping_prompt_checkpoint_path,
        num_loops=num_loops,
        use_residual=use_residual,
        device=device
    )
    model.to(device)
    model.eval()

    print(f"\nLoading {dataset_name} test dataset...")
    test_dataset = get_parse_then_hop_test_dataset(dataset_name)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    results = []
    total_em = 0
    total_f1 = 0
    total_samples = 0

    print("Evaluating...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            # Dataset returns (questions, complete_paths)
            # But for hopping evaluation, we use incomplete paths as input
            questions, complete_paths = batch

            # For this test, we feed the incomplete path (what parsing would output)
            # and check if the model completes it correctly
            # We need to create incomplete paths from complete paths
            incomplete_paths = []
            for path in complete_paths:
                parts = [p.strip() for p in path.split(';')]
                # Keep: e1 ; r1 ; e2 ; r2 (drop last answer)
                if len(parts) >= 5:
                    incomplete = ' ; '.join(parts[:-1])
                else:
                    incomplete = path
                incomplete_paths.append(incomplete)

            inputs = tokenizer(
                incomplete_paths,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )

            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256,
                num_beams=5,
                early_stopping=True
            )

            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for i, (pred, complete_path) in enumerate(zip(predictions, complete_paths)):
                pred_answer = extract_answer(pred)
                true_answer = extract_answer(complete_path)

                em = 1 if exact_match_score(pred_answer, true_answer) else 0
                f1 = f1_score(pred_answer, true_answer)[0]

                total_em += em
                total_f1 += f1
                total_samples += 1

                results.append({
                    'incomplete_path': incomplete_paths[i],
                    'true_path': complete_path,
                    'predicted_path': pred,
                    'true_answer': true_answer,
                    'predicted_answer': pred_answer,
                    'em': em,
                    'f1': f1
                })

            if (batch_idx + 1) % 50 == 0:
                print(f"Batch {batch_idx + 1}: EM={total_em/total_samples:.4f}, F1={total_f1/total_samples:.4f}")

    final_em = total_em / total_samples
    final_f1 = total_f1 / total_samples

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Internal Loops: {num_loops}")
    print(f"Use Residual: {use_residual}")
    print(f"Total Samples: {total_samples}")
    print(f"Exact Match: {final_em:.4f} ({final_em*100:.2f}%)")
    print(f"F1 Score: {final_f1:.4f} ({final_f1*100:.2f}%)")
    print(f"{'='*60}\n")

    # Save detailed results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, sep=';')
    print(f"Detailed results saved to: {output_file}")

    # Save summary
    summary = {
        'dataset': dataset_name,
        'num_loops': num_loops,
        'use_residual': use_residual,
        'total_samples': total_samples,
        'exact_match': final_em,
        'f1_score': final_f1
    }
    summary_file = output_file.replace('.csv', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    return final_em, final_f1


def compare_loop_counts(
    dataset_name: str,
    knit5_checkpoint_path: str,
    hopping_prompt_checkpoint_path: str,
    loop_range: list = [1, 2, 3, 4],
    use_residual: bool = True,
    batch_size: int = 8
):
    """Compare different numbers of internal loops"""
    comparison_results = []

    for num_loops in loop_range:
        print(f"\n{'#'*60}")
        print(f"Testing with {num_loops} internal loop(s)...")
        print(f"{'#'*60}\n")

        output_file = f"internal_loop_{dataset_name}_loops{num_loops}.csv"

        em, f1 = evaluate(
            dataset_name=dataset_name,
            knit5_checkpoint_path=knit5_checkpoint_path,
            hopping_prompt_checkpoint_path=hopping_prompt_checkpoint_path,
            num_loops=num_loops,
            use_residual=use_residual,
            batch_size=batch_size,
            output_file=output_file
        )

        comparison_results.append({
            'num_loops': num_loops,
            'em_score': em,
            'f1_score': f1
        })

    print(f"\n{'='*60}")
    print("LOOP COUNT COMPARISON")
    print(f"{'='*60}")
    for r in comparison_results:
        print(f"  {r['num_loops']} loop(s): EM={r['em_score']:.4f}, F1={r['f1_score']:.4f}")
    print(f"{'='*60}\n")

    # Save comparison
    df = pd.DataFrame(comparison_results)
    df.to_csv(f"internal_loop_comparison_{dataset_name}.csv", index=False)

    return comparison_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Internal Looped Hyperbolic Model')

    parser.add_argument('--dataset', type=str, default='wikimultihop')
    parser.add_argument('--knit5_checkpoint_path', type=str, required=True)
    parser.add_argument('--hopping_prompt_checkpoint_path', type=str, required=True)
    parser.add_argument('--num_loops', type=int, default=1)
    parser.add_argument('--use_residual', action='store_true', default=True)
    parser.add_argument('--no_residual', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--compare_loops', action='store_true')
    parser.add_argument('--loop_range', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--output_file', type=str, default='internal_loop_results.csv')

    args = parser.parse_args()

    use_residual = not args.no_residual

    if args.compare_loops:
        compare_loop_counts(
            dataset_name=args.dataset,
            knit5_checkpoint_path=args.knit5_checkpoint_path,
            hopping_prompt_checkpoint_path=args.hopping_prompt_checkpoint_path,
            loop_range=args.loop_range,
            use_residual=use_residual,
            batch_size=args.batch_size
        )
    else:
        evaluate(
            dataset_name=args.dataset,
            knit5_checkpoint_path=args.knit5_checkpoint_path,
            hopping_prompt_checkpoint_path=args.hopping_prompt_checkpoint_path,
            num_loops=args.num_loops,
            use_residual=use_residual,
            batch_size=args.batch_size,
            output_file=args.output_file
        )
