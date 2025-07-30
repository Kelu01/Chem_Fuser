import torch
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from model import DiffusionModel, MaskingScheduler
from vocab import MolData, Vocabulary
from torch.optim.lr_scheduler import ExponentialLR
import os
from vocab import extract_gz_if_needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

smiles_path = extract_gz_if_needed("data/canonical_smiles.txt.gz")


def diffusiontrain():
    # Initialize wandb for experiment tracking
    wandb.init(project="Chem_Fuser-v4", config={
        "batch_size": 128,
        "learning_rate": 0.0006,
        "num_epochs": 8,
        "scheduler": "linear"
    })

    config = wandb.config

    # Load vocabulary and dataset
    voc = Vocabulary("data/voc.txt")
    print(f"Vocabulary size: {len(voc)}")
    wandb.config.vocab_size = len(voc)

    dataset = MolData("data/canonical_smiles.txt", voc)
    data = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                      drop_last=True, collate_fn=MolData.collate_fn)

    # Initialize model
    model = DiffusionModel(voc)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    wandb.watch(model)

    # Parameter count logging
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    wandb.config.total_params = total_params

    # Optimizer and schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    mask_scheduler = MaskingScheduler(
        total_steps=config.num_epochs * len(data), schedule=config.scheduler)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

    os.makedirs("checkpoints", exist_ok=True)

    best_loss = float('inf')
    print("Starting Diffusion Training...")

    global_step = 0
    for epoch in range(1, config.num_epochs + 1):
        for batch in tqdm(data, total=len(data)):
            seqs = batch.long().to(device)
            t = torch.full((seqs.size(0),), mask_scheduler.get_t(global_step), device=device)

            # If using DataParallel, access the underlying model
            model_ref = model.module if isinstance(model, torch.nn.DataParallel) else model
            loss, n_masked = model_ref.likelihood(seqs, t)

            # Skip unstable or degenerate batches
            if not torch.isfinite(loss):
                print(f"⚠️ Skipping NaN loss at step {global_step}")
                global_step += 1
                continue
            if n_masked == 0:
                global_step += 1
                continue

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item(), "epoch": epoch, "step": global_step})

            # Periodically sample and evaluate unmasking quality
            if global_step % 500 == 0 and global_step != 0:
                masked_smiles, unmasked_smiles = model_ref.sample(
                    voc, seqs, mask_scheduler, current_step=global_step)
                valid = 0
                for masked, unmasked in zip(masked_smiles, unmasked_smiles):
                    print(f"Partially masked: {masked}")
                    print(f"Unmasked        : {unmasked}\n")
                    if Chem.MolFromSmiles(unmasked):
                        valid += 1

                valid_percent = 100 * valid / len(unmasked_smiles)
                print(f"Valid unmasked SMILES: {valid_percent:.1f}%\n")
                wandb.log({"unmasked_valid_%": valid_percent})

                # Save best-performing model
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    checkpoint_path = "checkpoints/chem_fuser.ckpt"
                    torch.save({
                        "model_state_dict": model_ref.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()
                    }, checkpoint_path)
                    wandb.save(checkpoint_path, base_path=os.path.dirname(checkpoint_path))
                    tqdm.write(f"🧠 New best model saved with loss: {best_loss:.2f}")

            global_step += 1

        # Step learning rate scheduler
        lr_scheduler.step()
        wandb.log({"learning_rate": lr_scheduler.get_last_lr()[0]})

        # Save final model after each epoch
        final_path = "checkpoints/chem_fuser_final.ckpt"
        torch.save({
            "model_state_dict": model_ref.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, final_path)
        wandb.save(final_path, base_path="checkpoints")

if __name__ == '__main__':
    diffusiontrain()