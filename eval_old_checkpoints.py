import torch
import pandas as pd
from transformers import AutoTokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import re
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def load_dataset():
    dev_df = pd.read_parquet("dataset/2wikimultihop/dev.parquet")
    return dev_df

def evaluate_checkpoint(checkpoint_path, model_name="t5-large"):
    dev_dataset = load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('base_model.'):
            new_state_dict[k.replace('base_model.', '')] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

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
            f1_val = f1_score(prediction, answer)

            total_em += em
            total_f1 += f1_val

    avg_em = (total_em / len(dev_dataset)) * 100
    avg_f1 = (total_f1 / len(dev_dataset)) * 100

    return avg_em, avg_f1

if __name__ == '__main__':
    results = []

    em, f1 = evaluate_checkpoint("/Users/fira/Desktop/DM and ML Lab/checkpoints/stage1_knowledge_integration_final.pt")
    results.append(["Stage 1: KI", f"{em:.2f}", f"{f1:.2f}"])

    em, f1 = evaluate_checkpoint("/Users/fira/Desktop/DM and ML Lab/checkpoints/stage2a_parsing_prompt_final.pt")
    results.append(["Stage 2a: Parsing", f"{em:.2f}", f"{f1:.2f}"])

    print(f"{'Stage':<25} {'EM (%)':<10} {'F1 (%)':<10}")
    print("-"*45)
    for row in results:
        print(f"{row[0]:<25} {row[1]:<10} {row[2]:<10}")
