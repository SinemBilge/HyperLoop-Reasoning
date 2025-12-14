# HyperLoop-Reasoning

Looped Hyperbolic Transformer for Multi-Hop Question Answering. Tests whether iterative hopping through hyperbolic space improves reasoning.

## Setup

```bash
pip install -r requirements.txt
```

## Required Checkpoints

Place in `checkpoints/` or use paths directly:

| Checkpoint | Description |
|------------|-------------|
| `knit5.pth` | Knowledge-integrated T5-Large (~3GB) |
| `parsing_prompt.pth` | Parsing soft prompt (~4MB) |
| `hopping_prompt.pth` | Hopping soft prompt (~4MB) |

## Evaluation

### Standard Hyperbolic (baseline, 1 hop)
```bash
python evaluate_looped_model.py \
    --dataset wikimultihop \
    --num_loops 1 \
    --batch_size 8 \
    --knit5_checkpoint_path checkpoints/knit5.pth \
    --parsing_prompt_checkpoint_path checkpoints/parsing_prompt.pth \
    --hopping_prompt_checkpoint_path checkpoints/hopping_prompt.pth
```

### Looped Hyperbolic (2+ loops)
```bash
python evaluate_looped_model.py \
    --dataset wikimultihop \
    --num_loops 2 \
    --batch_size 8 \
    --knit5_checkpoint_path checkpoints/knit5.pth \
    --parsing_prompt_checkpoint_path checkpoints/parsing_prompt.pth \
    --hopping_prompt_checkpoint_path checkpoints/hopping_prompt.pth
```

### Compare Loop Counts (1-4)
```bash
python evaluate_looped_model.py \
    --dataset wikimultihop \
    --compare_loops \
    --loop_range 1 2 3 4 \
    --batch_size 8 \
    --knit5_checkpoint_path checkpoints/knit5.pth \
    --parsing_prompt_checkpoint_path checkpoints/parsing_prompt.pth \
    --hopping_prompt_checkpoint_path checkpoints/hopping_prompt.pth
```

## Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--num_loops` | Number of hopping iterations | 2 |
| `--aggregation_method` | `last`, `weighted_avg`, `attention` | last |
| `--batch_size` | Batch size for evaluation | 8 |
| `--compare_loops` | Run comparison across loop counts | False |
| `--loop_range` | Loop counts to compare | 1 2 3 4 |

## Output

- `looped_model_results.csv` - Per-sample predictions
- `looped_model_results_summary.json` - EM score summary
- `loop_comparison_wikimultihop.csv` - Comparison across loop counts (if `--compare_loops`)

## Architecture

```
Question -> [Parse] -> Parsed Path -> [Hop] -> [Hop] -> ... -> Answer
                                       ^__________|
                                       feedback loop
```

The parsing and hopping models use hyperbolic layers. Looping applies the hopping step N times, feeding each output back as input.
