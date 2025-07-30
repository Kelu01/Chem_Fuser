"""
Optimized training script for Chem_Fuser using Automatic Mixed Precision (AMP)
and torch.compile() for faster training on modern GPUs (PyTorch 2.0+).

This version wraps the DiffusionModel with torch.compile() for graph-level optimizations,
and uses torch.cuda.amp.autocast() with GradScaler for efficient mixed precision training.

Recommended for Ampere or newer GPUs with CUDA >= 11.6 and PyTorch >= 2.0.
Fallbacks to standard training(train.py) if run on unsupported hardware.

Note:
- AMP accelerates training and reduces memory usage.
- torch.compile() enables dynamic graph optimization during runtime.
"""

import torch
from torch.optim.lr_scheduler import ExponentialLR
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
from model import DiffusionModel
from vocab import MolData, Vocabulary
from model import MaskingScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def diffusiontrain():
    wandb.init(project="Chem_Fuser-v4", config={
        "batch_size": 128,
        "learning_rate": 0.0006,
        "num_epochs": 12,
        "scheduler": "linear"
    })

    config = wandb.config

    voc = Vocabulary("data/voc.txt.txt")
    print(f"Vocabulary size: {len(voc)}")
    wandb.config.vocab_size = len(voc)

    dataset = MolData("data/canonical_smiles.txt", voc)
    data = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=MolData.collate_fn)

    model = DiffusionModel(voc).to(device)

    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"⚠️ torch.compile() failed: {e}")

    wandb.watch(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    wandb.config.total_params = total_params

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = MaskingScheduler(total_steps=config.num_epochs * len(data), schedule=config.scheduler)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    best_loss = float('inf')
    print("Starting Diffusion Training...")

    global_step = 0
    for epoch in range(1, config.num_epochs + 1):
        for batch in tqdm(data, total=len(data)):
            seqs = batch.long().to(device)
            t = torch.full((seqs.size(0),), scheduler.get_t(global_step), device=device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                loss, n_masked = model.likelihood(seqs, t)

            if n_masked == 0:
                global_step += 1
                continue

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            wandb.log({"loss": loss.item(), "epoch": epoch, "step": global_step})

            if global_step % 500 == 0 and global_step != 0:
                masked_smiles, unmasked_smiles = model.sample(voc, seqs, scheduler, current_step=global_step)
                valid = 0
                for masked, unmasked in zip(masked_smiles, unmasked_smiles):
                    print(f"Partially masked: {masked}")
                    print(f"Unmasked        : {unmasked}\n")
                    if Chem.MolFromSmiles(unmasked):
                        valid += 1

                valid_percent = 100 * valid / len(unmasked_smiles)
                print(f"Valid unmasked SMILES: {valid_percent:.1f}%\n")
                wandb.log({"unmasked_valid_%": valid_percent})

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    checkpoint_path = "checkpoints/chem_fuser.ckpt"
                    torch.save(model.state_dict(), checkpoint_path)
                    wandb.save(checkpoint_path, base_path=os.path.dirname(checkpoint_path))
                    tqdm.write(f"New best model saved with loss: {best_loss:.4f}")

            global_step += 1

        lr_scheduler.step()
        wandb.log({"learning_rate": lr_scheduler.get_last_lr()[0]})
        torch.save(model.state_dict(), "checkpoints/chem_fuser_final.ckpt")
        wandb.save("checkpoints/chem_fuser_final.ckpt", base_path="checkpoints")

if __name__ == '__main__':
    diffusiontrain()