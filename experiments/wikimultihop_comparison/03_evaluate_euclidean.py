"""
Experiment 3: Evaluate Stage 2 Euclidean (Full Pipeline)
Evaluates T5 after KI training + Euclidean parsing + hopping prompts
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.util import load_dataset
from transformers import AutoTokenizer
from src.models import T5ModelWithAdditionalLayer
from src.config import Config
import torch
from tqdm import tqdm
from src.eval import exact_match_score, f1_score
import csv
from datetime import datetime
import glob

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in directory"""
    checkpoints = glob.glob(f"{checkpoint_dir}/**/*.pth", recursive=True)
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    latest = max(checkpoints, key=os.path.getmtime)
    print(f"Using checkpoint: {latest}")
    return latest

def evaluate_euclidean():
    print("="*60)
    print("Experiment 3: Stage 2 Euclidean (Full Pipeline)")
    print("="*60)

    config = Config()
    model_name = config.t5_model.model_name

    # Load dataset
    print(f"\n[1/5] Loading dataset...")
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset(
        "../../dataset/2wikimultihop",
        do_correct_wrong_evidences=True
    )

    # Load model
    print(f"[2/5] Loading {model_name} with Euclidean layer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Find checkpoints
    print("[3/5] Finding checkpoints...")
    ki_checkpoint = find_latest_checkpoint("checkpoints/stage1_ki")

    # Check if hopping checkpoint exists (final stage)
    hopping_dir = "checkpoints/stage2_euclidean_hopping"
    if os.path.exists(hopping_dir):
        hopping_checkpoint = find_latest_checkpoint(hopping_dir)
        print(f"Using hopping checkpoint: {hopping_checkpoint}")
        final_checkpoint = hopping_checkpoint
    else:
        print("⚠️  Hopping checkpoint not found, using KI checkpoint only")
        final_checkpoint = ki_checkpoint

    model = T5ModelWithAdditionalLayer(
        layer_type='euclidean',
        curvature=1.0,
        checkpoint_hyperbolic_knit5=final_checkpoint,
        with_model_state_dict=True,
        gpu_parallelization=False
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    print(f"Using device: {device}")
    print(f"Dev set size: {len(dev_dataset)}")

    # Evaluate
    print("\n[4/5] Evaluating on dev set (used as test)...")
    total_em = 0
    total_f1 = 0

    with torch.no_grad():
        for idx, row in tqdm(dev_dataset.iterrows(), total=len(dev_dataset), desc="Evaluating"):
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

    print("\n[5/5] Results:")
    print(f"Exact Match (EM): {avg_em:.2f}%")
    print(f"F1 Score: {avg_f1:.2f}%")

    # Save results
    results_file = "results.csv"
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Exp3_Stage2_Euclidean',
            model_name,
            'Stage 2: Parse + Hop',
            'Euclidean',
            f'{avg_em:.2f}',
            f'{avg_f1:.2f}',
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ])

    print(f"\n✓ Results saved to {results_file}")
    return avg_em, avg_f1

if __name__ == '__main__':
    evaluate_euclidean()
