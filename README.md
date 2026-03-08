# HyperLoop-Reasoning

Looped Hyperbolic Transformer for Multi-Hop Question Answering.

This project extends T5 with a **hyperbolic additional layer** (Poincaré ball) and **weight-shared looping** (LoopedHyperbolicLayer) for multi-hop reasoning. Training follows a two-stage soft prompt approach: (1) parsing questions into sub-questions, and (2) hopping across knowledge graph entities.

> **Acknowledgement:** This repository builds upon [HyperbolicMultiHopReasoning](https://github.com/caisa-lab/HyperbolicMultiHopReasoning) by the CAISA Lab.

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── src/
│   ├── config.py                        # Training configurations
│   ├── eval.py                          # Evaluation metrics (EM, F1)
│   ├── knowledge_graph.py               # Knowledge graph construction
│   ├── models/
│   │   ├── hyperbolic_t5_additional_layer.py   # T5 + configurable additional layer
│   │   ├── hyperbolic_t5_with_looping_alpha_i.py # T5 + alpha input injection looping
│   │   ├── hyperbolic_model_utils.py           # HyperbolicLayer & LoopedHyperbolicLayer
│   │   ├── soft_prompt_model.py                # Soft prompt wrapper
│   │   └── hyperbolic_nn_resnet/               # Poincaré/Lorentz manifold math
│   ├── datasets/                        # Dataset loaders (WikiMultiHop, MetaQA, MLPQ, PQL)
│   ├── train/
│   │   ├── soft_prompt_trainer.py       # Soft prompt training loop (DDP-ready)
│   │   ├── soft_prompt_trainer_alpha_i.py # Trainer with alpha_i parameter handling
│   │   └── model_trainer.py             # Base model trainer
│   └── utils/
├── train_knowledge_integration.py       # Stage 1: Knowledge integration
├── train_parse_then_hop.py              # Stage 2a: Parsing prompt training
├── train_random_walk.py                 # Stage 2b: Hopping prompt training
├── test_parsing.py                      # Test parsing stage
├── test_random_walk.py                  # Test hopping stage
├── test_parse_then_hop.py               # Test full pipeline (parse → hop)
├── analyze_loop_intermediates.py        # Per-iteration representation analysis
├── compute_distance_accuracy.py         # Hyperbolic vs Euclidean distance ordering
├── compute_delta_hyperbolicity.py       # δ-hyperbolicity per layer
└── requirements.txt
```

## Required Checkpoints

| Checkpoint | Description | Used in |
|------------|-------------|---------|
| `knit5.pth` | Knowledge-integrated T5 (~3GB) | Training & Testing |
| `parsing_prompt.pth` | Parsing soft prompt (~4MB) | Testing |
| `hopping_prompt.pth` | Hopping soft prompt (~4MB) | Testing |

## Training

The additional layer type is controlled via `--additional_layer` (`identity`, `euclidean`, `hyperbolic`) and looping via `--num_loop_iterations`.

### Stage 2a: Parsing
```bash
python train_parse_then_hop.py \
    --additional_layer hyperbolic \
    --dataset 2wikimultihop \
    --knit5_checkpoint_path checkpoints/knit5.pth \
    --num_loop_iterations 2 \
    --curvature 1.0 \
    --epochs 50
```

### Stage 2b: Hopping
```bash
python train_random_walk.py \
    --additional_layer hyperbolic \
    --dataset 2wikimultihop \
    --knit5_checkpoint_path checkpoints/knit5.pth \
    --num_loop_iterations 2 \
    --curvature 1.0 \
    --epochs 50
```

## Testing

### Full pipeline (parse → hop)
```bash
python test_parse_then_hop.py \
    --dataset 2wikimultihop \
    --additional_layer_parse hyperbolic \
    --additional_layer_hop hyperbolic \
    --knit5_checkpoint_path checkpoints/knit5.pth \
    --parsing_prompt_checkpoint_path checkpoints/parsing_prompt.pth \
    --hopping_prompt_checkpoint_path checkpoints/hopping_prompt.pth \
    --batch_size 8
```

## Analysis Scripts

### Inference-time loop analysis
Analyzes per-iteration intermediate representations to test whether iteration *t* resolves hop *t*:
```bash
python analyze_loop_intermediates.py \
    --dataset 2wikimultihop \
    --prompt_checkpoint checkpoints/hopping_prompt.pth \
    --knit5_checkpoint checkpoints/knit5.pth \
    --num_loop_iterations 2 \
    --max_samples 500
```

### Distance ordering accuracy
Evaluates whether hyperbolic distances produce correct entity ordering along multi-hop paths:
```bash
python compute_distance_accuracy.py \
    --dataset 2wikimultihop \
    ...
```

### δ-hyperbolicity measurement
Measures how much hyperbolic structure exists in each layer's representations:
```bash
python compute_delta_hyperbolicity.py \
    --dataset 2wikimultihop \
    ...
```

## Architecture

```
                    Looped Hyperbolic Layer
                    ======================
Input -> T5 Encoder -> [HyperbolicLayer + iteration embedding + residual] x T -> T5 Decoder -> Output
                              ^                                            |
                              |____________________________________________|
                                        (weight-shared, T iterations)
```

The `LoopedHyperbolicLayer` applies a single shared `HyperbolicLayer` *T* times with:
- Learned iteration embeddings (one per timestep)
- Residual connections at each iteration
- Minimal parameter overhead (~T × hidden_dim)

### Alpha Input Injection Variant

An alternative looping strategy using learnable per-loop mixing coefficients (`T5ModelWithLoopedHyperbolic`):

```
h_i = alpha_i * f(h_{i-1}) + (1 - alpha_i) * h_encoder
```

Each `alpha_i` is a learned scalar (passed through sigmoid) controlling how much of the transformed representation vs. the original encoder output is retained at each iteration. Includes Poincaré ball clamping for hyperbolic stability. See `src/models/hyperbolic_t5_with_looping_alpha_i.py`.
