# MiniQwen: Lightweight Qwen Model with Multi-Stage Training

A compact, efficient implementation of Mini Qwen architecture with Mixture of Experts (MoE), designed for code generation and terminal reasoning tasks. This project implements knowledge distillation from larger Qwen models to create a lightweight variant suitable for resource-constrained environments.

## Project Status

Current Phase: Development

This project is actively under development and has been made public for the purpose of running the training 

## Overview

MiniQwen is a distilled version of Qwen3 that incorporates:
- Mixture of Experts (MoE) architecture for efficient computation
- Knowledge distillation from Qwen2.5 teacher models
- Grouped Query Attention (GQA) for reduced memory footprint
- RoPE (Rotary Position Embeddings) for improved position encoding
- Optimized for code generation and shell command understanding

## Architecture

### Model Components
- **Attention Mechanism**: Multi-head attention with GQA support
- **MLP Layer**: Mixture of Experts with configurable expert count
- **Normalization**: RMSNorm for stable training
- **Position Encoding**: Rotary Position Embeddings (RoPE)

### Default Configuration
```
Embedding Dimension: 512
Attention Heads: 8
KV Heads: 4
Layers: 6
MLP Hidden Size: 2048
Number of Experts: 8
Experts per Token: 2
MoE Intermediate Size: 1024
```

## Training Pipeline

### Stage 1: Knowledge Distillation
Distill knowledge from a larger Qwen2.5 teacher model using:
- Cross-entropy loss for supervised learning
- KL divergence loss for distribution matching
- Temperature scaling for soft targets

### Stage 2: Supervised Fine-Tuning (SFT)
Fine-tune on task-specific datasets for:
- Python code generation
- Shell command synthesis
- Terminal operation understanding

### Stage 3: RLHF (Planned)
Reinforcement Learning from Human Feedback for:
- Code quality improvement
- Safety and reliability
- User preference alignment

## Installation

```bash
# Clone the repository
git clone https://github.com/aymen-000/MiniQwen.git
cd MiniQwen

# Install dependencies
```

## Usage

### Training (Knowledge Distillation)

```bash
python DistillMiniQwen/distill_train.py \
    --teacher_model Qwen/Qwen2.5-0.5B \
    --dataset jtatman/python-code-dataset-500k \
    --batch_size 4 \
    --num_epochs 3 \
    --max_samples 10000 \
    --output_dir checkpoints
```

### Testing Trained Model

```bash
# Run benchmark tests
python test_model.py --checkpoint qwen3_moe_distilled_best.pt

# Interactive mode
python test_model.py --checkpoint qwen3_moe_distilled_best.pt --interactive

# Custom prompts
python test_model.py \
    --checkpoint qwen3_moe_distilled_best.pt \
    --prompts "Instruction: Write a Python function to reverse a string\nAnswer:"
```

### Key Arguments

**Model Architecture:**
- `--n_embed`: Embedding dimension (default: 512)
- `--n_head`: Number of attention heads (default: 8)
- `--n_layer`: Number of transformer layers (default: 6)
- `--num_experts`: Number of MoE experts (default: 8)

**Training:**
- `--batch_size`: Training batch size (default: 4)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--temperature`: Distillation temperature (default: 2.0)
- `--alpha_ce`: Weight for cross-entropy loss (default: 0.5)
- `--alpha_kd`: Weight for KL divergence loss (default: 0.5)

**Dataset:**
- `--dataset`: HuggingFace dataset name
- `--max_samples`: Maximum training samples
- `--max_length`: Maximum sequence length (default: 256)

## Features

### Implemented
- Qwen3 MoE architecture
- Knowledge distillation training
- Grouped Query Attention (GQA)
- RoPE position embeddings
- RMSNorm layer normalization
- Model checkpointing
- Inference with sampling strategies

### In Development
- Supervised Fine-Tuning (SFT) pipeline
- RLHF training loop
- Multi-dataset training support
- Model quantization (INT8/INT4)
- Efficient expert routing optimization

### Planned
- GGUF format export for llama.cpp
- ONNX export for cross-platform deployment
- Treminal interface
- API serving endpoints

## Model Performance

Performance metrics will be updated as development progresses.


## Technical Details

### Knowledge Distillation
The distillation process uses a combination of:
- Hard targets: Cross-entropy loss with ground truth labels
- Soft targets: KL divergence loss with teacher predictions
- Temperature scaling: Softens probability distributions for better knowledge transfer

### MoE Architecture
- Dynamic expert routing based on input tokens
- Top-k expert selection per token
- Load balancing across experts
- Efficient sparse computation

## Known Issues

- MoE implementation computes all experts (optimization in progress)
- Generation quality varies with temperature and sampling parameters
- Large vocabulary size may cause memory constraints on smaller GPUs

## Contributing

This project is currently in development. Contributions will be welcomed once the repository is made public.

## License

License information will be provided upon public release.

## Acknowledgments

- Based on the Qwen architecture by Alibaba Cloud
- Inspired by knowledge distillation techniques from DistilBERT and TinyBERT
- Uses datasets from HuggingFace Hub

## Contact

For questions or collaboration inquiries, please open an issue once this repo is completed.
For collaboration in this pahze contact me  : aymne011@gmail.com

## Roadmap

- [ ] Complete knowledge distillation pipeline
- [ ] Implement SFT training
- [ ] Add RLHF training loop
- [ ] Optimize MoE expert routing
- [ ] Add model quantization
- [ ] Create deployment documentation
- [ ] Benchmark against baseline models
- [ ] Public release

---

**Note:** This project is under active development. Features, APIs, and documentation are subject to change.