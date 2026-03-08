"""
Analyze per-iteration intermediate representations from LoopedHyperbolicLayer.

For each sample, computes distances (Euclidean and hyperbolic geodesic) between
the source entity and hop-1/hop-2 entities at each loop iteration. This tests
the hypothesis: does iteration 1 primarily resolve hop 1, and iteration 2 resolve hop 2?

Usage:
    python analyze_loop_intermediates.py \
        --dataset 2wikimultihop \
        --prompt_checkpoint <path_to_T2_soft_prompt.pth> \
        --knit5_checkpoint <path_to_knit5.pth> \
        --dataset_type dev \
        --num_loop_iterations 2 \
        --max_samples 500
"""

import argparse
import json
import os
from math import exp, log

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config import Config
from src.models import SoftPromptModel, T5ModelWithAdditionalLayer
from src.models.hyperbolic_model_utils import LoopedHyperbolicLayer
from src.utils.util import load_dataset, load_train_test_pql_dataset
from src.knowledge_graph import (
    create_knowledge_graph_wikimultihop,
    create_knowledge_graph_metaqa,
    create_knowledge_graph_mlpq,
    create_knowledge_graph_pql,
)
from src.datasets import (
    RandomWalkMetaQADataset,
    RandomWalkMLPQDataset,
    RandomWalkWikiHopDataset,
    RandomWalkPQLDataset,
)


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def hyperbolic_distance(u, v, c=1.0):
    u, v = np.array(u), np.array(v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    diff_norm = np.linalg.norm(u - v)
    intermediate = 2 * c * (diff_norm ** 2) / ((1 - c * norm_u ** 2) * (1 - c * norm_v ** 2))
    arg = max(1 + intermediate, 1.0)
    return (1 / np.sqrt(c)) * np.arccosh(arg)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def mean_pool_semicolon_text(text, hidden_tensor, tokenizer):
    """Mean-pool subword embeddings per semicolon-separated unit."""
    units = [u.strip() for u in text.split(';')]
    num_units = len(units)
    hidden_np = hidden_tensor[0].cpu().numpy()  # (seq_len, hidden_dim)
    enc_inputs = tokenizer(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(enc_inputs["input_ids"][0])

    unit_embeddings = [[] for _ in range(num_units)]
    current_unit_idx = 0
    for i, tok in enumerate(tokens):
        tok_str = tok.replace("▁", "").strip()
        if tok_str == ";":
            if current_unit_idx < num_units - 1:
                current_unit_idx += 1
            continue
        if i < hidden_np.shape[0]:
            unit_embeddings[current_unit_idx].append(hidden_np[i])

    pooled, actual_units = [], []
    for u_idx in range(num_units):
        if unit_embeddings[u_idx]:
            pooled.append(np.mean(unit_embeddings[u_idx], axis=0))
            actual_units.append(units[u_idx])
    return pooled, actual_units


def get_dataset(dataset_name, dataset_type):
    """Load dataset (reuses same logic as compute_distance_accuracy.py)."""
    if dataset_name in ['2wikimultihop', 'wikimultihop', '2wikihop', 'wikihop']:
        train_ds, dev_ds, test_ds, _, _, _ = load_dataset(
            'dataset/2wikimultihop', do_correct_wrong_evidences=True
        )
        all_data = pd.concat([train_ds, dev_ds, test_ds])
        kg = create_knowledge_graph_wikimultihop(all_data)
        rw_train = RandomWalkWikiHopDataset(kg, dev_ds, test_ds, steps=3, type='train')
        rw_dev = RandomWalkWikiHopDataset(kg, dev_ds, test_ds, steps=3, type='dev')
        rw_test = RandomWalkWikiHopDataset(kg, dev_ds, test_ds, steps=3, type='test')
    elif dataset_name == 'metaqa':
        df_dev = pd.read_json("dataset/metaqa/2hops/qa_dev_evidences.json")
        df_train = pd.read_json("dataset/metaqa/2hops/qa_train_evidences.json")
        df_test = pd.read_json("dataset/metaqa/2hops/qa_test_evidences.json")
        df_kg = pd.concat([df_dev, df_train, df_test])
        kg = create_knowledge_graph_metaqa(df_kg, from_kb=False, max_answers=1)
        rw_train = RandomWalkMetaQADataset(kg, df_dev, df_test, steps=3, type='train')
        rw_dev = RandomWalkMetaQADataset(kg, df_dev, df_test, steps=3, type='dev')
        rw_test = RandomWalkMetaQADataset(kg, df_dev, df_test, steps=3, type='test')
    elif dataset_name == 'mlpq':
        train_df = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_train_question_evidences.json', lines=True)
        val_df = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_dev_question_evidences.json', lines=True)
        test_df = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_test_question_evidences.json', lines=True)
        df_kg = pd.concat([train_df, val_df, test_df])
        kg = create_knowledge_graph_mlpq(df_kg, from_kb=False)
        rw_train = RandomWalkMLPQDataset(kg, val_df, test_df, steps=3, type='train')
        rw_dev = RandomWalkMLPQDataset(kg, val_df, test_df, steps=3, type='dev')
        rw_test = RandomWalkMLPQDataset(kg, val_df, test_df, steps=3, type='test')
    elif dataset_name == 'pql':
        file_path = "dataset/pathquestion/PQ-2H.txt"
        train, val, test = load_train_test_pql_dataset(file_path, random_state=789)
        kg = create_knowledge_graph_pql(file_path, from_kb=False)
        rw_train = RandomWalkPQLDataset(kg, val, test, steps=3, type='train')
        rw_dev = RandomWalkPQLDataset(kg, val, test, steps=3, type='dev')
        rw_test = RandomWalkPQLDataset(kg, val, test, steps=3, type='test')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if dataset_type == 'train':
        return rw_train
    elif dataset_type == 'dev':
        return rw_dev
    elif dataset_type == 'test':
        return rw_test
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


def load_model(knit5_checkpoint, prompt_checkpoint, num_loop_iterations):
    """Load T5 + LoopedHyperbolicLayer + soft prompt from checkpoints."""
    soft_prompt_ckpt = torch.load(prompt_checkpoint, map_location='cpu')
    soft_prompt = nn.Parameter(soft_prompt_ckpt['soft_prompt_state_dict'])
    additional_layer_sd = soft_prompt_ckpt['additional_linear_layer']

    model = T5ModelWithAdditionalLayer(
        layer_type='hyperbolic',
        checkpoint_hyperbolic_knit5=knit5_checkpoint,
        num_loop_iterations=num_loop_iterations,
    )
    model.additional_layer.load_state_dict(additional_layer_sd, strict=False)
    soft_model = SoftPromptModel(model, None, soft_prompt=soft_prompt)
    return soft_model


def run_with_intermediates(model, tokenizer, input_text, label_text, device):
    """
    Run a single sample through the model and collect intermediate representations
    from each loop iteration.

    Returns:
        intermediates: list of tensors [h^(0), h^(1), ..., h^(T)], each (1, seq_len, hidden_dim)
        curvature: float, the learned curvature value
    """
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    labels = tokenizer(label_text, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    label_ids = labels.input_ids.to(device)

    net = model
    # Prepend soft prompt (same as SoftPromptModel._forward_soft_prompt_only_enc)
    soft_prompt_input = net.soft_prompt.unsqueeze(0).expand(input_ids.size(0), -1, -1).to(device)
    input_embeddings = net.knit5.shared(input_ids)
    concat_embeddings = torch.cat([soft_prompt_input, input_embeddings], dim=1)
    soft_prompt_mask = torch.ones((1, soft_prompt_input.size(1)), device=device)
    concat_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)

    # Run encoder
    encoder_outputs = net.knit5.encoder(
        inputs_embeds=concat_embeddings,
        attention_mask=concat_mask,
        return_dict=True,
    )
    hidden_states = encoder_outputs.last_hidden_state

    # Get intermediates from the looped layer
    additional_layer = net.knit5.additional_layer
    if isinstance(additional_layer, LoopedHyperbolicLayer):
        _, intermediates = additional_layer(hidden_states, return_intermediates=True)
    else:
        # Single-pass (T=1): just input and output
        output = additional_layer(hidden_states)
        intermediates = [hidden_states.clone(), output.clone()]

    curvature = 0.0
    if hasattr(additional_layer, 'manifold'):
        curvature = additional_layer.manifold.c.item()
    elif hasattr(additional_layer, 'hyperbolic_layer') and hasattr(additional_layer.hyperbolic_layer, 'manifold'):
        curvature = additional_layer.hyperbolic_layer.manifold.c.item()

    return intermediates, curvature


def analyze(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Config()
    tokenizer = AutoTokenizer.from_pretrained(config.t5_model.model_name)

    print(f"Loading dataset: {args.dataset} ({args.dataset_type})")
    dataset = get_dataset(args.dataset, args.dataset_type)

    print(f"Loading model with {args.num_loop_iterations} loop iterations...")
    model = load_model(args.knit5_checkpoint, args.prompt_checkpoint, args.num_loop_iterations)
    model.to(device)
    model.eval()

    # Get curvature for hyperbolic distance
    c_readable = 0.0
    hl = model.knit5.additional_layer
    if hasattr(hl, 'manifold'):
        c_readable = log(exp(hl.manifold.c.item()) + 1.0)
    elif hasattr(hl, 'hyperbolic_layer') and hasattr(hl.hyperbolic_layer, 'manifold'):
        c_readable = log(exp(hl.hyperbolic_layer.manifold.c.item()) + 1.0)
    print(f"Curvature (readable): {c_readable:.4f}")

    max_samples = min(args.max_samples, len(dataset))
    num_iterations = args.num_loop_iterations

    # Storage for per-iteration distance results
    # For each sample: distances from source entity to each other entity at each iteration
    results = []

    print(f"Analyzing {max_samples} samples...")
    with torch.no_grad():
        for idx in tqdm(range(max_samples), desc="Analyzing intermediates"):
            input_text, label_text = dataset[idx]

            try:
                intermediates, curvature = run_with_intermediates(
                    model, tokenizer, input_text, label_text, device
                )
            except Exception as e:
                print(f"Skipping sample {idx}: {e}")
                continue

            # The label_text has the complete path: "e1 ; r1 ; e2 ; r2 ; e3"
            # We analyze distances using the label_text structure on intermediates
            # But intermediates are encoder representations of the INPUT (incomplete path)
            # So we pool using input_text: "e1 ; r1 ; r2"

            sample_result = {
                'idx': idx,
                'input': input_text,
                'label': label_text,
                'iterations': []
            }

            for t, h_t in enumerate(intermediates):
                # h_t includes soft prompt tokens at the front; strip them
                soft_prompt_len = model.soft_prompt.size(0)
                h_t_no_prompt = h_t[:, soft_prompt_len:, :]  # (1, input_seq_len, hidden_dim)

                pooled, units = mean_pool_semicolon_text(input_text, h_t_no_prompt, tokenizer)

                if len(pooled) < 2:
                    continue

                source = pooled[0]
                iter_distances = {
                    'iteration': t,  # 0 = before any loop, 1 = after iter 0, etc.
                    'units': units,
                    'euclidean_from_source': [],
                    'hyperbolic_from_source': [],
                    'cosine_from_source': [],
                }
                for j in range(1, len(pooled)):
                    iter_distances['euclidean_from_source'].append(
                        float(euclidean_distance(source, pooled[j]))
                    )
                    if c_readable > 0:
                        try:
                            iter_distances['hyperbolic_from_source'].append(
                                float(hyperbolic_distance(source, pooled[j], c=c_readable))
                            )
                        except Exception:
                            iter_distances['hyperbolic_from_source'].append(float('nan'))
                    iter_distances['cosine_from_source'].append(
                        float(cosine_similarity(source, pooled[j]))
                    )

                sample_result['iterations'].append(iter_distances)

            results.append(sample_result)

    # --- Aggregate statistics ---
    print("\n" + "=" * 80)
    print("PER-ITERATION DISTANCE ANALYSIS")
    print("=" * 80)

    # For 2-hop: input has 3 units (entity, relation1, relation2)
    # Distances from source to: relation1 (idx 0), relation2 (idx 1)
    for t in range(num_iterations + 1):
        euclid_dists = {j: [] for j in range(5)}  # up to 5 units after source
        hyp_dists = {j: [] for j in range(5)}
        cos_sims = {j: [] for j in range(5)}

        for r in results:
            for it in r['iterations']:
                if it['iteration'] == t:
                    for j, d in enumerate(it['euclidean_from_source']):
                        euclid_dists[j].append(d)
                    for j, d in enumerate(it['hyperbolic_from_source']):
                        hyp_dists[j].append(d)
                    for j, d in enumerate(it['cosine_from_source']):
                        cos_sims[j].append(d)

        label = "Before loop" if t == 0 else f"After iteration {t}"
        print(f"\n--- {label} (t={t}) ---")
        for j in range(5):
            if euclid_dists[j]:
                n = len(euclid_dists[j])
                e_mean = np.mean(euclid_dists[j])
                e_std = np.std(euclid_dists[j])
                c_mean = np.mean(cos_sims[j])
                c_std = np.std(cos_sims[j])
                unit_label = f"unit {j+1}"
                print(f"  Source -> {unit_label}: "
                      f"Euclidean={e_mean:.4f}±{e_std:.4f}, "
                      f"Cosine={c_mean:.4f}±{c_std:.4f} "
                      f"(n={n})")
                if hyp_dists[j]:
                    valid = [x for x in hyp_dists[j] if not np.isnan(x)]
                    if valid:
                        h_mean = np.mean(valid)
                        h_std = np.std(valid)
                        print(f"                     "
                              f"Hyperbolic={h_mean:.4f}±{h_std:.4f}")

    # --- Cosine similarity between consecutive iterations ---
    print(f"\n--- Inter-iteration cosine similarity (representation change) ---")
    for t in range(1, num_iterations + 1):
        cos_changes = []
        for r in results:
            iters = {it['iteration']: it for it in r['iterations']}
            if t - 1 in iters and t in iters:
                # Use all euclidean distances as a proxy for representation content
                prev_dists = iters[t - 1]['euclidean_from_source']
                curr_dists = iters[t]['euclidean_from_source']
                if prev_dists and curr_dists and len(prev_dists) == len(curr_dists):
                    # Compare the distance profiles
                    change = np.mean([abs(a - b) for a, b in zip(prev_dists, curr_dists)])
                    cos_changes.append(change)
        if cos_changes:
            print(f"  Iteration {t-1} -> {t}: mean distance change = {np.mean(cos_changes):.4f}±{np.std(cos_changes):.4f}")

    # Save detailed results
    output_path = args.output or f"loop_intermediates_{args.dataset}_{args.dataset_type}_T{args.num_loop_iterations}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze per-iteration intermediates from LoopedHyperbolicLayer')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['2wikimultihop', 'wikimultihop', 'metaqa', 'mlpq', 'pql'])
    parser.add_argument('--prompt_checkpoint', type=str, required=True,
                        help='Path to soft prompt checkpoint (.pth)')
    parser.add_argument('--knit5_checkpoint', type=str, required=True,
                        help='Path to base knit5 checkpoint (.pth)')
    parser.add_argument('--dataset_type', type=str, default='dev',
                        choices=['train', 'dev', 'test'])
    parser.add_argument('--num_loop_iterations', type=int, default=2)
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: auto-generated)')
    args = parser.parse_args()
    analyze(args)
