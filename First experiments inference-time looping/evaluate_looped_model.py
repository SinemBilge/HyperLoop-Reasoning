import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
import pandas as pd
from looped_transformer_model import create_looped_model
from src.datasets import get_parse_then_hop_test_dataset
import json


def evaluate_looped_model(
    dataset_name: str,
    knit5_checkpoint_path: str,
    parsing_prompt_checkpoint_path: str,
    hopping_prompt_checkpoint_path: str,
    num_loops: int = 2,
    aggregation_method: str = 'last',
    batch_size: int = 8,
    output_file: str = 'looped_model_results.csv'
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained('google/t5-large-lm-adapt')

    print(f"Creating looped model with {num_loops} loops.")
    looped_model = create_looped_model(
        knit5_checkpoint_path=knit5_checkpoint_path,
        parsing_prompt_checkpoint_path=parsing_prompt_checkpoint_path,
        hopping_prompt_checkpoint_path=hopping_prompt_checkpoint_path,
        additional_layer_type='hyperbolic',
        curvature=0.37,
        num_loops=num_loops,
        aggregation_method=aggregation_method,
        device=device
    )
    looped_model.to(device)
    looped_model.eval()

    print(f"\nLoading {dataset_name} test dataset...")
    test_dataset = get_parse_then_hop_test_dataset(dataset_name)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    results = []
    total_em = 0
    total_samples = 0

    print("Evaluating on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            # Dataset returns (questions, complete_paths) tuples
            questions, complete_paths = batch

            # Extract final answer from complete_path: "e1 ; r1 ; e2 ; r2 ; answer"
            answers = [path.split(';')[-1].strip() for path in complete_paths]

            inputs = tokenizer(
                list(questions),
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )

            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            predictions, intermediate = looped_model.evaluate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=tokenizer
            )

            for i, (pred, answer, complete_path) in enumerate(zip(predictions, answers, complete_paths)):
                # Extract answer from prediction path
                pred_parts = [p.strip() for p in pred.split(';')]
                pred_answer = pred_parts[-1] if pred_parts else pred

                pred_answer = pred_answer.lower().strip()
                answer_norm = answer.lower().strip()

                em = 1 if pred_answer == answer_norm else 0
                total_em += em
                total_samples += 1

                results.append({
                    'question': questions[i],
                    'true_answer': answer,
                    'true_path': complete_path,
                    'predicted_answer': pred_answer,
                    'full_prediction': pred,
                    'parsed_path': intermediate['parsed_path'][i],
                    'loop_outputs': [loop['decoded_paths'][i]
                                   for loop in intermediate['loop_outputs']],
                    'em_score': em,
                    'num_loops': num_loops
                })

            if (batch_idx + 1) % 10 == 0:
                current_em = total_em / total_samples
                print(f"Batch {batch_idx + 1}, Current EM: {current_em:.4f}")

    final_em = total_em / total_samples
    print(f"\n{'='*50}")
    print(f"Final Results:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Num Loops: {num_loops}")
    print(f"  Aggregation: {aggregation_method}")
    print(f"  Total Samples: {total_samples}")
    print(f"  Exact Match: {final_em:.4f}")
    print(f"{'='*50}\n")

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, sep=';')
    print(f"Detailed results saved to: {output_file}")

    summary = {
        'dataset': dataset_name,
        'num_loops': num_loops,
        'aggregation_method': aggregation_method,
        'total_samples': total_samples,
        'exact_match': final_em
    }

    summary_file = output_file.replace('.csv', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    return final_em, results


def compare_loop_counts(
    dataset_name: str,
    knit5_checkpoint_path: str,
    parsing_prompt_checkpoint_path: str,
    hopping_prompt_checkpoint_path: str,
    loop_range: list = [1, 2, 3, 4],
    batch_size: int = 8
):
    comparison_results = []

    for num_loops in loop_range:
        print(f"\n{'#'*60}")
        print(f"Evaluating with {num_loops} loop(s)...")
        print(f"{'#'*60}\n")

        output_file = f"looped_results_{dataset_name}_loops{num_loops}.csv"

        em_score, _ = evaluate_looped_model(
            dataset_name=dataset_name,
            knit5_checkpoint_path=knit5_checkpoint_path,
            parsing_prompt_checkpoint_path=parsing_prompt_checkpoint_path,
            hopping_prompt_checkpoint_path=hopping_prompt_checkpoint_path,
            num_loops=num_loops,
            aggregation_method='last',
            batch_size=batch_size,
            output_file=output_file
        )

        comparison_results.append({
            'num_loops': num_loops,
            'em_score': em_score
        })

    print(f"\n{'='*60}")
    print("Loop Count Comparison:")
    print(f"{'='*60}")
    for result in comparison_results:
        print(f"  {result['num_loops']} loop(s): EM = {result['em_score']:.4f}")
    print(f"{'='*60}\n")

    comparison_df = pd.DataFrame(comparison_results)
    comparison_file = f"loop_comparison_{dataset_name}.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Comparison saved to: {comparison_file}")

    return comparison_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Looped Hyperbolic Transformer')

    parser.add_argument('--dataset', type=str, default='wikimultihop')
    parser.add_argument('--knit5_checkpoint_path', type=str, required=True)
    parser.add_argument('--parsing_prompt_checkpoint_path', type=str, required=True)
    parser.add_argument('--hopping_prompt_checkpoint_path', type=str, required=True)
    parser.add_argument('--num_loops', type=int, default=2)
    parser.add_argument('--aggregation_method', type=str, default='last',
                       choices=['last', 'weighted_avg', 'attention'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--compare_loops', action='store_true')
    parser.add_argument('--loop_range', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--output_file', type=str, default='looped_model_results.csv')

    args = parser.parse_args()

    if args.compare_loops:
        compare_loop_counts(
            dataset_name=args.dataset,
            knit5_checkpoint_path=args.knit5_checkpoint_path,
            parsing_prompt_checkpoint_path=args.parsing_prompt_checkpoint_path,
            hopping_prompt_checkpoint_path=args.hopping_prompt_checkpoint_path,
            loop_range=args.loop_range,
            batch_size=args.batch_size
        )
    else:
        evaluate_looped_model(
            dataset_name=args.dataset,
            knit5_checkpoint_path=args.knit5_checkpoint_path,
            parsing_prompt_checkpoint_path=args.parsing_prompt_checkpoint_path,
            hopping_prompt_checkpoint_path=args.hopping_prompt_checkpoint_path,
            num_loops=args.num_loops,
            aggregation_method=args.aggregation_method,
            batch_size=args.batch_size,
            output_file=args.output_file
        )
