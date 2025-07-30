import numpy as np
import torch
import re
import numpy as np
from torch.utils.data import Dataset
import os
import gzip
import shutil

def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)

class Vocabulary(object):
    """A class for handling encoding/decoding from SMILES to an array of indices"""
    def __init__(self, file, max_length=140):
        self.special_tokens = ["[PAD]","[GO]"]
        self.clusters = self.init_from_file(file)
        self.chars = self.special_tokens + self.clusters
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        

    def encode(self, char_list):
        """Takes a list of characters (e.g. '[NH]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.vocab[char]
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        matrix = [int(i) for i in matrix]
        chars = []
        for i in matrix:
            if i == self.vocab["[EOS]"]:
                break
            if i == self.vocab["[GO]"]:
                continue
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        smiles = self.restore_halogen(smiles)
        return smiles

    def tokenize(self, smiles):
        """Takes a SMILES and returns a list of characters/tokens"""
        regex = r'(\[[^\[\]]{1,6}\])'
        smiles = self.replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        tokenized = ['[GO]']
        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        tokenized.append("[EOS]")
        return tokenized
    
    def replace_halogen(self, smiles):
        """Replaces two-character halogens with single characters for simpler tokenization."""
        return smiles.replace('Cl', 'L').replace('Br', 'R')

    def restore_halogen(self, smiles):
        """Restores halogens back to their proper form after decoding."""
        return smiles.replace('L', 'Cl').replace('R', 'Br')

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
        return chars

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return "Vocabulary containing {} tokens: {}".format(len(self), self.chars)

    
class MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing SMILES.

        Args:
                fname : path to a file containing \n separated SMILES.
                voc   : a Vocabulary instance

        Returns:
                A custom PyTorch dataset for training the Prior.
    """
    def __init__(self, fname, voc):
        self.voc = voc
        self.smiles = []
        with open(fname, 'r') as f:
            for line in f:
                self.smiles.append(line.split()[0])

    def __getitem__(self, i):
        mol = self.smiles[i]
        tokenized = self.voc.tokenize(mol)
        encoded = self.voc.encode(tokenized)
        return Variable(encoded)

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))

    @classmethod

    def collate_fn(cls, arr):
        """Function to take a list of encoded sequences and turn them into a batch"""
        max_length = max([seq.size(0) for seq in arr])
        collated_arr = Variable(torch.zeros(len(arr), max_length))
        for i, seq in enumerate(arr):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr
    
def extract_gz_if_needed(filepath):
    """
    Extracts a .gz file if it exists and the extracted .txt version is missing.
    Returns the path to the extracted .txt file.
    """
    if filepath.endswith('.gz'):
        extracted_path = filepath[:-3]  # remove .gz extension
        if not os.path.exists(extracted_path):
            print(f"Extracting {filepath} to {extracted_path}...")
            with gzip.open(filepath, 'rb') as f_in, open(extracted_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return extracted_path
    return filepath