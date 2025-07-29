import re
def replace_halogen(string):
    """Regex to replace Br and Cl with single letters (optional if needed)."""
    string = re.sub(r'Br', 'R', string)
    string = re.sub(r'Cl', 'L', string)
    return string

def construct_vocabulary(smiles_path):
    """Builds vocabulary from canonicalized SMILES and writes to 'Voc.txt'."""
    add_chars = set()
    with open(smiles_path, 'r') as file:
        for line in file:
            smiles = line.strip()
            regex = r'(\[[^\[\]]{1,6}\])'
            smiles = replace_halogen(smiles)
            char_list = re.split(regex, smiles)
            for char in char_list:
                if char.startswith('['):
                    add_chars.add(char)
                else:
                    [add_chars.add(unit) for unit in char]

    # Add special tokens
    add_chars.update(["[MASK]", "[PAD]", "[EOS]", "[CLS]", "[SEP]", "[UNK]"] + [f"[unused{i}]" for i in range(1, 6)])
    
    # Write vocabulary to file (optional)
    with open('data/Voc.txt', 'w') as f:
        for char in sorted(add_chars):
            f.write(f"{char}\n")

    print(f"Number of characters in vocabulary (including [MASK]): {len(add_chars)}")

if __name__ == "__main__":
    print("Constructing vocabulary...")
    smiles_list = "data/canonical_smiles.txt"
    voc_chars = construct_vocabulary(smiles_list)