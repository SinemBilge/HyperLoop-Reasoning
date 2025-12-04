"""
Experiment 2: Evaluate Euclidean (Identity Layer) Trained Model
"""

import sys
sys.path.append('../..')

from src.utils.util import load_dataset
from transformers import AutoTokenizer
from src.models import T5ModelWithAdditionalLayer
import torch
from tqdm import tqdm
from src.eval import exact_match_score, f1_score
import csv
import os
from datetime import datetime
import glob

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in directory"""
    checkpoints = glob.glob(f"{checkpoint_dir}/**/knit5*.pth", recursive=True)
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    # Get most recent
    latest = max(checkpoints, key=os.path.getmtime)
    print(f"Using checkpoint: {latest}")
    return latest

def evaluate_euclidean():
    print("="*50)
    print("Experiment 2: Euclidean (Identity Layer)")
    print("="*50)

    # Load dataset
    print("\n[1/4] Loading dataset...")
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset(
        "../../dataset/2wikimultihop",
        do_correct_wrong_evidences=True
    )

    # Load model
    print("[2/4] Loading trained model with identity layer...")
    tokenizer = AutoTokenizer.from_pretrained("google/t5-large-lm-adapt")

    # Find checkpoint
    checkpoint_path = find_latest_checkpoint("checkpoints/euclidean")

    model = T5ModelWithAdditionalLayer(
        layer_type='identity',
        curvature=0.0,
        checkpoint_hyperbolic_knit5=checkpoint_path,
        with_model_state_dict=True,
        gpu_parallelization=False
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    print(f"Using device: {device}")
    print(f"Dev set size: {len(dev_dataset)}")

    # Evaluate
    print("\n[3/4] Evaluating on dev set...")
    total_em = 0
    total_f1 = 0

    with torch.no_grad():
        for idx, row in tqdm(dev_dataset.iterrows(), total=len(dev_dataset)):
            question = row['question']
            answer = row['answer']

            input_text = f"question: {question}"
            inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512).to(device)

            outputs = model.generate(**inputs, max_length=50)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

            em = 1 if exact_match_score(prediction, answer) else 0
            f1 = f1_score(prediction, answer)[0]

            total_em += em
            total_f1 += f1

    avg_em = (total_em / len(dev_dataset)) * 100
    avg_f1 = (total_f1 / len(dev_dataset)) * 100

    print("\n[4/4] Results:")
    print(f"Exact Match (EM): {avg_em:.2f}%")
    print(f"F1 Score: {avg_f1:.2f}%")

    # Save results
    results_file = "results.csv"
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            '02_euclidean',
            'T5-large-lm-adapt + KI',
            'Identity',
            f'{avg_em:.2f}',
            f'{avg_f1:.2f}',
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ])

    print(f"\n✓ Results saved to {results_file}")
    return avg_em, avg_f1

if __name__ == '__main__':
    evaluate_euclidean()
