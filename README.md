# Chem_Fuser

Chem_Fuser is a deep generative model for molecular structure recovery using a diffusion-based architecture. It learns to reconstruct complete SMILES strings from partially masked sequences, enabling chemically valid molecule generation via denoising.

This project is implemented in PyTorch and is designed for flexibility, performance, and chemical validity during training.

---

## Features

- Transformer-based diffusion model for SMILES denoising
- Custom vocabulary and forward masking scheduler
- Automatic handling of valid SMILES evaluation during sampling
- Multi-GPU support with PyTorch `DataParallel`
- Optional AMP + `torch.compile()` training for performance
- Integrated experiment tracking via Weights & Biases (wandb)

---
## Installation

1. Clone the repository:

```bash
git clone https://github.com/Kelu01/Chem_Fuser.git
cd chem_fuser
```

2. Install dependencies:
   
```bash
pip install -r requirements.txt
```

---

## Training

- Standard Training (with optional DataParallel)

```bash
python train.py
```

- Optimized Training (AMP + torch.compile)
```bash
python train_amp.py
```

