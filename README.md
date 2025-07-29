# Chem_Fuser

Chem_Fuser is a deep generative model for molecular structure recovery using a diffusion-based architecture. It learns to reconstruct complete SMILES strings from partially masked sequences, enabling chemically valid molecule generation via denoising.

This project is implemented in PyTorch and is designed for flexibility, performance, and chemical validity during training.

> **Note:** This repository is an improved and extended version of [Chem_Fuser on Hugging Face](https://huggingface.co/kelu01/Chem_Fuser), developed by the same author.

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

---

## Project Layout and Directory Overview

The following directories and files contain the key components of the Chem_Fuser project:

- `model.py` – Core model implementation, including the transformer-based diffusion architecture and sampling logic.
- `vocab.py` – SMILES vocabulary management and tokenization functions.
- `train.py` – Main training script supporting both single-GPU and multi-GPU training using `DataParallel`.
- `train_amp.py` – Optional training script that uses Automatic Mixed Precision (AMP) and `torch.compile()` for optimized performance on supported hardware.
- `data/` – Input resources:
  - `voc.txt` – Vocabulary tokens used for SMILES tokenization.
  - `canonical_smiles.txt` – Dataset of SMILES strings (one per line).
- `checkpoints/` – Directory where model weights are saved during training.
- `inference/` – Inference pipeline and related files:
  - `inference.ipynb` – Notebook for evaluating the trained model on masked SMILES.
  - `process.py` – Utility script for preparing input and running inference.
  - `test_set.txt`, `masked_test_set.txt`, `inference_data.csv` – Inference input and output files.
- `preprocessing/` – Scripts for preparing the dataset and vocabulary:
  - `create_dataset.py` – Processes raw SMILES into training-ready format.
  - `create_voc.py` – Builds the vocabulary file from the dataset.
- `LICENSE` – MIT license file.
- `README.md` – Project overview, usage guide, and documentation.
- `requirements.txt` – Python dependencies required to run the project.
