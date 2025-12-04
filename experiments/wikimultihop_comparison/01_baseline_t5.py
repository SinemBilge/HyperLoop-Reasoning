"""
Experiment 1: Baseline T5 Performance (No Training)
Evaluates pretrained T5 on WikiMultiHop dev set
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.util import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import Config
import torch
from tqdm import tqdm
from src.eval import exact_match_score, f1_score
import csv
from datetime import datetime

def evaluate_baseline():
    print("="*60)
    print("Experiment 1: Baseline T5 (No Training)")
    print("="*60)

    config = Config()
    model_name = config.t5_model.model_name

    # Load dataset
    print(f"\n[1/4] Loading dataset...")
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset(
        "../../dataset/2wikimultihop",
        do_correct_wrong_evidences=True
    )

    # Load model
    print(f"[2/4] Loading pretrained {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    print(f"Using device: {device}")
    print(f"Dev set size: {len(dev_dataset)}")

    # Evaluate
    print("\n[3/4] Evaluating on dev set (used as test)...")
    total_em = 0
    total_f1 = 0

    with torch.no_grad():
        for idx, row in tqdm(dev_dataset.iterrows(), total=len(dev_dataset), desc="Evaluating"):
            question = row['question']
            answer = row['answer']

            # Prepare input
            input_text = f"question: {question}"
            inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512).to(device)

            # Generate
            outputs = model.generate(**inputs, max_length=50)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Score
            em = 1 if exact_match_score(prediction, answer) else 0
            f1 = f1_score(prediction, answer)[0]

            total_em += em
            total_f1 += f1

    # Calculate metrics
    avg_em = (total_em / len(dev_dataset)) * 100
    avg_f1 = (total_f1 / len(dev_dataset)) * 100

    print("\n[4/4] Results:")
    print(f"Exact Match (EM): {avg_em:.2f}%")
    print(f"F1 Score: {avg_f1:.2f}%")

    # Save results
    results_file = "results.csv"
    file_exists = os.path.exists(results_file)

    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Experiment', 'Model', 'Stage', 'Additional Layer', 'EM (%)', 'F1 (%)', 'Timestamp'])

        writer.writerow([
            'Exp1_Baseline',
            model_name,
            'Pretrained (no training)',
            'None',
            f'{avg_em:.2f}',
            f'{avg_f1:.2f}',
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ])

    print(f"\n✓ Results saved to {results_file}")
    return avg_em, avg_f1

if __name__ == '__main__':
    evaluate_baseline()
