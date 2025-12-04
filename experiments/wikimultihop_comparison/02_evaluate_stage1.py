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
    checkpoints = glob.glob(f"{checkpoint_dir}", recursive=True)
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    latest = max(checkpoints, key=os.path.getmtime)
    print(f"Using checkpoint: {latest}")
    return latest

def evaluate_stage1():
    config = Config()
    model_name = config.t5_model.model_name

    print("\n[1/4] Loading dataset...")
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset(
        "dataset/2wikimultihop",
        do_correct_wrong_evidences=True
    )

    print("[2/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    checkpoint_path = find_latest_checkpoint("experiments/wikimultihop_comparison/checkpoints/stage1_ki/Nov15_23-43-21_AdaFactor_knowledge_integration_bsize64_lr0.001_identity_c1.0/knit5.pth")

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

    dev_subset = dev_dataset.head(16)
    print(f"Using device: {device}")
    print(f"Dev set size: {len(dev_subset)}")

    print("\n[3/4] Evaluating...")
    total_em = 0
    total_f1 = 0

    output_file = "stage1_predictions.txt"
    with open(output_file, 'w') as f:
        with torch.no_grad():
            for idx, row in tqdm(dev_subset.iterrows(), total=len(dev_subset), desc="Evaluating"):
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

                f.write(f"Question: {question}\n")
                f.write(f"Answer: {answer}\n")
                f.write(f"New answer: {prediction}\n\n")

    avg_em = (total_em / len(dev_subset)) * 100
    avg_f1 = (total_f1 / len(dev_subset)) * 100

    print("\n[4/4] Results:")
    print(f"Exact Match (EM): {avg_em:.2f}%")
    print(f"F1 Score: {avg_f1:.2f}%")

    results_file = "results.csv"
    print(f"\n✓ Results saved to {results_file}")

if __name__ == '__main__':
    evaluate_stage1()
