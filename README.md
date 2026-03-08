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
      --dataset 2wikimultihop\
      --additional_layer_parse hyperbolic \
      --additional_layer_hop hyperbolic \  
      --knit5_checkpoint_path checkpoints/knit5.pth \
      --parsing_prompt_checkpoint_path checkpoints/parsing_prompt.pth \
      --hopping_prompt_checkpoint_path checkpoints/hopping_prompt.pth \
      --num_loop_iterations_parse 1 \
      --num_loop_iterations_hop 2 \
      --batch_size 8   
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


