"""
Experiment 5: Evaluate Looped Transformer (Optional)
Evaluates looped T5 (k⊗L architecture) on WikiMultiHop
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.util import load_dataset
from transformers import AutoTokenizer
from src.config import Config
import torch
from tqdm import tqdm
from src.eval import exact_match_score, f1_score
import csv
from datetime import datetime
import glob

# Try to import looped model
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../looped_transformer_experiments'))
    from models.looped_t5 import LoopedT5
    from models.soft_prompt_looped import SoftPromptLoopedModel
    LOOPED_AVAILABLE = True
except ImportError:
    LOOPED_AVAILABLE = False

def find_latest_checkpoint(checkpoint_dir, pattern="*.pth"):
    """Find the latest checkpoint in directory"""
    checkpoints = glob.glob(f"{checkpoint_dir}/**/{pattern}", recursive=True)
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    latest = max(checkpoints, key=os.path.getmtime)
    print(f"Using checkpoint: {latest}")
    return latest

def evaluate_looped():
    print("="*60)
    print("Experiment 5: Looped Transformer (k⊗L)")
    print("="*60)

    if not LOOPED_AVAILABLE:
        print("\n❌ Looped transformer models not available!")
        print("Make sure looped_transformer_experiments/ exists.")
        return None, None

    config = Config()
    model_name = config.t5_model.model_name

    # Load dataset
    print(f"\n[1/5] Loading dataset...")
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset(
        "../../dataset/2wikimultihop",
        do_correct_wrong_evidences=True
    )

    # Check if looped checkpoints exist
    looped_ki_dir = "../../looped_transformer_experiments/checkpoints/looped/knowledge_integration"

    if not os.path.exists(looped_ki_dir):
        print("\n⚠️  No looped transformer checkpoints found!")
        print("\nTo train looped transformer, follow these steps:")
        print("\n1. Train Stage 1 (Knowledge Integration):")
        print("   cd looped_transformer_experiments")
        print("   python train_looped_knowledge_integration.py \\")
        print("       --dataset wikimultihop \\")
        print("       --k 6 --L 4 \\")
        print("       --epochs 50 --batch_size 64 \\")
        print("       --learning_rate 0.001 \\")
        print("       --checkpoint_save_path checkpoints/looped/knowledge_integration/")
        print("\n2. Then rerun this evaluation script")
        return None, None

    # Load model
    print(f"[2/5] Loading looped T5 model (k=6, L=4)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Find checkpoint
    print("[3/5] Finding checkpoints...")
    try:
        ki_checkpoint = find_latest_checkpoint(looped_ki_dir, "*.pth")
    except ValueError as e:
        print(f"\n❌ {e}")
        print("Train the looped transformer first (see instructions above)")
        return None, None

    # Load checkpoint
    checkpoint = torch.load(ki_checkpoint, map_location='cpu')

    # Initialize looped model
    model = LoopedT5(model_name=model_name, k=6, L=4)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

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

            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=50
            )
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
            'Exp5_Looped_Transformer',
            f'{model_name} + Looped(k=6,L=4)',
            'Stage 1: KI (Looped)',
            'Looped k=6⊗L=4',
            f'{avg_em:.2f}',
            f'{avg_f1:.2f}',
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ])

    print(f"\n✓ Results saved to {results_file}")
    return avg_em, avg_f1

if __name__ == '__main__':
    evaluate_looped()
