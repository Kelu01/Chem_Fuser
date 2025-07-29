import torch
import random
from tqdm import tqdm
from rdkit import Chem
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vocab import Vocabulary, MolData
from model import forward_diffusion
from torch.nn import functional as F

voc = Vocabulary("data/voc.txt")
pad_token = voc.vocab["[PAD]"]
mask_token = voc.vocab["[MASK]"]
go_token = voc.vocab["[GO]"]
eos_token = voc.vocab["[EOS]"]

# Load 10,000 valid SMILES
with open("inference/test_set.txt", "r") as f:
    smiles_list = [line.strip() for line in f if line.strip()]

assert len(smiles_list) == 10_000

# Tokenize
encoded = []
for smi in smiles_list:
    tokens = voc.encode(voc.tokenize(smi))
    # Remove [GO] and [EOS]
    if tokens[0] == go_token:
        tokens = tokens[1:]
    if tokens[-1] == eos_token:
        tokens = tokens[:-1]
    encoded.append(torch.tensor(tokens, dtype=torch.long))

max_len = max(len(seq) for seq in encoded)
padded = [F.pad(seq, (0, max_len - len(seq)), value=pad_token) for seq in encoded]
tensor_batch = torch.stack(padded)  # (10000, max_len)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_batch = tensor_batch.to(device)

# Generate random t ∈ [0.15, 0.7] per sequence
t_vals = torch.tensor([random.uniform(0.15, 0.7) for _ in range(tensor_batch.size(0))], device=device)
x_masked, _ = forward_diffusion(tensor_batch, t_vals, mask_token, pad_token) # Apply forward diffusion (masking)

# Decode masked SMILES
def strip_pad(seq):
    return [i for i in seq if i != pad_token]

masked_smiles = [voc.decode(strip_pad(x_masked[i].tolist())) for i in range(x_masked.size(0))]

with open("inference/masked_test_set.txt", "w") as f:
    for smi in masked_smiles:
        f.write(smi + "\n")

print("Masked SMILES saved to masked_test_set.txt")