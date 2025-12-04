# WikiMultiHop Systematic Comparison

## Goal
Compare T5-base performance across different architectures on 2WikiMultiHopQA

## Experimental Pipeline

### Experiment 1: Baseline T5 (No Training)
Test vanilla pretrained T5-base without any fine-tuning

```bash
cd experiments/wikimultihop_comparison
python 01_baseline_t5.py
```

**Time:** ~5-10 minutes

---

### Experiment 2: Knowledge Integration (Stage 1)
Train T5-base on knowledge graph triples (identity layer = standard Euclidean)

```bash
python train_knowledge_integration.py \
  --dataset wikimultihop \
  --epochs 50 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --checkpoint_save_path experiments/wikimultihop_comparison/checkpoints/stage1_ki \
  --tboard_logs_save_path experiments/wikimultihop_comparison/logs/stage1_ki \
  --additional_layer identity

# Evaluate
cd experiments/wikimultihop_comparison
python 02_evaluate_stage1.py
```

**Time:** ~15-25 hours (T5-base) vs ~75 hours (T5-large)

---

### Experiment 3: Stage 2 Euclidean (Parsing + Hopping)
Train soft prompts with Euclidean layer on top of Stage 1 checkpoint

**3a. Parsing Prompt:**
```bash
python ../../train_parse_then_hop.py \
  --additional_layer euclidean \
  --learning_rate 0.3 \
  --curvature 1.0 \
  --dataset wikimultihop \
  --epochs 50 \
  --batch_size 64 \
  --knit5_checkpoint_path experiments/wikimultihop_comparison/checkpoints/stage1_ki/YOUR_CHECKPOINT.pth \
  --checkpoint_save_path experiments/wikimultihop_comparison/checkpoints/stage2_euclidean_parsing \
  --tboard_logs_save_path experiments/wikimultihop_comparison/logs/stage2_euclidean_parsing \
  --train_parsing
```

**3b. Hopping Prompt:**
```bash
python ../../train_parse_then_hop.py \
  --additional_layer euclidean \
  --learning_rate 0.3 \
  --curvature 1.0 \
  --dataset wikimultihop \
  --epochs 50 \
  --batch_size 64 \
  --knit5_checkpoint_path experiments/wikimultihop_comparison/checkpoints/stage1_ki/YOUR_CHECKPOINT.pth \
  --parsing_prompt_checkpoint_path experiments/wikimultihop_comparison/checkpoints/stage2_euclidean_parsing/YOUR_PARSING.pth \
  --checkpoint_save_path experiments/wikimultihop_comparison/checkpoints/stage2_euclidean_hopping \
  --tboard_logs_save_path experiments/wikimultihop_comparison/logs/stage2_euclidean_hopping \
  --train_hopping
```

**Evaluate:**
```bash
cd experiments/wikimultihop_comparison
python 03_evaluate_euclidean.py
```

**Time:** ~4-6 hours each (parsing + hopping)

---

### Experiment 4: Stage 2 Hyperbolic (Parsing + Hopping)
Train soft prompts with Hyperbolic layer on top of Stage 1 checkpoint

**4a. Parsing Prompt:**
```bash
python ../../train_parse_then_hop.py \
  --additional_layer hyperbolic \
  --learning_rate 0.3 \
  --curvature 0.37 \
  --dataset wikimultihop \
  --epochs 50 \
  --batch_size 64 \
  --knit5_checkpoint_path experiments/wikimultihop_comparison/checkpoints/stage1_ki/YOUR_CHECKPOINT.pth \
  --checkpoint_save_path experiments/wikimultihop_comparison/checkpoints/stage2_hyperbolic_parsing \
  --tboard_logs_save_path experiments/wikimultihop_comparison/logs/stage2_hyperbolic_parsing \
  --train_parsing
```

**4b. Hopping Prompt:**
```bash
python ../../train_parse_then_hop.py \
  --additional_layer hyperbolic \
  --learning_rate 0.3 \
  --curvature 0.37 \
  --dataset wikimultihop \
  --epochs 50 \
  --batch_size 64 \
  --knit5_checkpoint_path experiments/wikimultihop_comparison/checkpoints/stage1_ki/YOUR_CHECKPOINT.pth \
  --parsing_prompt_checkpoint_path experiments/wikimultihop_comparison/checkpoints/stage2_hyperbolic_parsing/YOUR_PARSING.pth \
  --checkpoint_save_path experiments/wikimultihop_comparison/checkpoints/stage2_hyperbolic_hopping \
  --tboard_logs_save_path experiments/wikimultihop_comparison/logs/stage2_hyperbolic_hopping \
  --train_hopping
```

**Evaluate:**
```bash
cd experiments/wikimultihop_comparison
python 04_evaluate_hyperbolic.py
```

**Time:** ~4-6 hours each (parsing + hopping)

---

### Experiment 5: Looped Transformer (Optional)
Compare with looped transformer architecture

```bash
cd experiments/wikimultihop_comparison
python 05_evaluate_looped.py
```

**Time:** TBD (depends on existing looped implementation)

---

## View Results

```bash
cd experiments/wikimultihop_comparison
python view_results.py
```

Shows table with all EM and F1 scores + improvements over baseline.

---

## Expected Results Structure

```
experiments/wikimultihop_comparison/
├── README.md                          # This file
├── results.csv                        # All experimental results
├── 01_baseline_t5.py                  # Baseline evaluation
├── 02_evaluate_stage1.py              # Stage 1 KI evaluation
├── 03_evaluate_euclidean.py           # Euclidean full pipeline eval
├── 04_evaluate_hyperbolic.py          # Hyperbolic full pipeline eval
├── 05_evaluate_looped.py              # Looped transformer (optional)
├── view_results.py                    # Display results table
├── checkpoints/
│   ├── stage1_ki/                     # KI checkpoints
│   ├── stage2_euclidean_parsing/      # Euclidean parsing
│   ├── stage2_euclidean_hopping/      # Euclidean hopping
│   ├── stage2_hyperbolic_parsing/     # Hyperbolic parsing
│   └── stage2_hyperbolic_hopping/     # Hyperbolic hopping
└── logs/                              # TensorBoard logs
```

---

## Timeline Estimate (T5-base)

| Stage | Task | Time |
|-------|------|------|
| Exp 1 | Baseline eval | 10 min |
| Exp 2 | Stage 1 KI | 15-25 hours |
| Exp 3a | Euclidean parsing | 2-3 hours |
| Exp 3b | Euclidean hopping | 2-3 hours |
| Exp 4a | Hyperbolic parsing | 2-3 hours |
| Exp 4b | Hyperbolic hopping | 2-3 hours |
| **Total** | | **~25-40 hours** |

With T5-large: ~70-100 hours total

---

## Model Configuration

To use T5-base (faster):
Edit `src/config.py` line 101:
```python
self.model_name = "google/t5-base-lm-adapt"
```

To use T5-large (paper setting):
```python
self.model_name = "google/t5-large-lm-adapt"  # Default
```

---

## Key Questions Being Answered

1. **Does KI training help?** (Exp 2 vs Exp 1)
2. **Does hyperbolic geometry help?** (Exp 4 vs Exp 3)
3. **How much improvement?** (View results table)
4. **Is looped transformer competitive?** (Exp 5 vs others)
