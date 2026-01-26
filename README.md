# HyperLoop-Reasoning

Looped Hyperbolic Transformer for Multi-Hop Question Answering.
First Experiments directory contains inference-time experiments with added looping.
Current directory has training scripts for integrated looping architecture in soft prompt training.

## Setup

```bash
pip install -r requirements.txt
```
## Required Checkpoints for Training with Soft Prompts

| Checkpoint | Description |
|------------|-------------|
| `knit5.pth` | Knowledge-integrated T5-Large (~3GB) |

## Required Checkpoints for Testing

| Checkpoint | Description |
|------------|-------------|
| `knit5.pth` | Knowledge-integrated T5-Large (~3GB) |
| `parsing_prompt.pth` | Parsing soft prompt (~4MB) |
| `hopping_prompt.pth` | Hopping soft prompt (~4MB) |


## Training with Looping

### 1. Hyperbolic Layer looping

Loops the hyperbolic layer output N types with input injection:

```
Encoder -> hidden_states -> [Hyperbolic + Input Injection] x N -> Decoder
```

#### Stage 2a: Parsing training
```bash
python train_parse_then_hop_with_looping.py \
        --additional_layer hyperbolic \
        --dataset wikimultihop \
        --knit5_checkpoint_path <path> \
        --num_loops 4 \
        --max_norm 0.9 \
        --epochs 50
```

#### Stage 2b: Hopping training
```bash
python train_random_walk_with_looping.py \
        --additional_layer hyperbolic \
        --dataset wikimultihop \
        --knit5_checkpoint_path <path> \
        --num_loops 4 \
        --max_norm 0.9 \
        --epochs 50
```


### 2. Euclidean Layer Looping

Similar to hyperbolic, only with looping happening in Euclidean layer:

#### Stage 2a: Parsing training
```bash
python train_parse_then_hop_with_looping_euclidean.py \
        --additional_layer linear \
        --dataset wikimultihop \
        --knit5_checkpoint_path <path> \
        --num_loops 2 \
        --epochs 50
```

#### Stage 2b: Hopping training
```bash
python train_random_walk_with_looping_euclidean.py \
        --additional_layer linear \
        --dataset wikimultihop \
        --knit5_checkpoint_path <path> \
        --num_loops 2 \
        --epochs 50
```

## Testing/Inferencing with Looping

### Hyperbolic Looped Architecture
```bash
python test_parse_then_hop_looped.py \
        --dataset metaqa \
        --additional_layer_parse hyperbolic \
        --additional_layer_hop hyperbolic \
        --knit5_checkpoint_path checkpoints/knowledge_integration/knit5.pth \
        --parsing_prompt_checkpoint_path checkpoints/parse/soft_prompt.pth \
        --hopping_prompt_checkpoint_path checkpoints/hop/soft_prompt.pth \
        --num_loops 4 \
        --batch_size 8
```

### Euclidean Looped Architecture
```bash
python test_parse_then_hop_looped_euclidean.py \
        --dataset metaqa \
        --knit5_checkpoint_path checkpoints/knowledge_integration/knit5.pth \
        --parsing_prompt_checkpoint_path checkpoints/wikimultihop/parse/euclidean_looped/soft_prompt.pth \
        --hopping_prompt_checkpoint_path checkpoints/wikimultihop/hop/euclidean_looped/soft_prompt.pth \
        --num_loops 4 \
        --batch_size 8
```

## Inference time looping (first experiments): Two Looping Approaches

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
