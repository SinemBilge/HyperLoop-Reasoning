# HyperLoop-Reasoning

Looped Hyperbolic Transformer for Multi-Hop Question Answering. Tests whether iterative computation in hyperbolic space improves reasoning.

## Setup

```bash
pip install -r requirements.txt
```

## Required Checkpoints

| Checkpoint | Description |
|------------|-------------|
| `knit5.pth` | Knowledge-integrated T5-Large (~3GB) |
| `parsing_prompt.pth` | Parsing soft prompt (~4MB) |
| `hopping_prompt.pth` | Hopping soft prompt (~4MB) |

## Two Looping Approaches

### 1. Internal Loop (Recommended)

Loops the hyperbolic layer transformation on encoder hidden states:

```
Encoder -> hidden_states -> [Hyperbolic x N] -> Decoder
```

**Advantages**: Uses same trained weights, no input/output format change.

```bash
# Single evaluation
python evaluate_internal_loop.py \
    --dataset wikimultihop \
    --num_loops 2 \
    --batch_size 8 \
    --knit5_checkpoint_path checkpoints/knit5.pth \
    --hopping_prompt_checkpoint_path checkpoints/hopping_prompt.pth

# Compare loop counts
python evaluate_internal_loop.py \
    --dataset wikimultihop \
    --compare_loops \
    --loop_range 1 2 3 4 \
    --batch_size 8 \
    --knit5_checkpoint_path checkpoints/knit5.pth \
    --hopping_prompt_checkpoint_path checkpoints/hopping_prompt.pth
```

Options:
- `--num_loops`: Number of hyperbolic layer iterations (default: 1)
- `--use_residual` / `--no_residual`: Use residual connections between loops
- `--compare_loops`: Test multiple loop counts

### 2. External Loop (Parse-then-Hop chain)

Loops the full generation, feeding output back as input:

```
Question -> [Parse] -> [Hop] -> [Hop] -> ... -> Answer
```

```bash
python evaluate_looped_model.py \
    --dataset wikimultihop \
    --num_loops 2 \
    --batch_size 8 \
    --knit5_checkpoint_path checkpoints/knit5.pth \
    --parsing_prompt_checkpoint_path checkpoints/parsing_prompt.pth \
    --hopping_prompt_checkpoint_path checkpoints/hopping_prompt.pth
```

## Output Files

- `*_results.csv` - Per-sample predictions with EM/F1
- `*_summary.json` - Aggregated metrics
- `*_comparison_*.csv` - Loop count comparison results

## Architecture Diagram

```
                    Internal Looping
                    ================
Input -> T5 Encoder -> [Hyperbolic Layer] -> T5 Decoder -> Output
                              ^     |
                              |_____|  (loop N times)


                    External Looping
                    ================
Question -> Parse Model -> Hop Model -> Hop Model -> ... -> Answer
                               ^            |
                               |____________|  (feed output as input)
```
