from rdkit import Chem
def canonicalize_smiles_from_file(fname):
    """Reads a SMILES file and returns a list of RDKIT SMILES"""
    with open(fname, 'r') as f:
        smiles_list = []
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("{} lines processed.".format(i))
            smiles = line.split(" ")[0]
            mol = Chem.MolFromSmiles(smiles)
            if filter_mol(mol):
                smiles_list.append(Chem.MolToSmiles(mol))
        print("{} SMILES retrieved".format(len(smiles_list)))
        return smiles_list

def filter_mol(mol, max_heavy_atoms=50, min_heavy_atoms=10, element_list=[6,7,8,9,16,17,35]):
    """Filters molecules on number of heavy atoms and atom types"""
    if mol is not None:
        num_heavy = min_heavy_atoms<mol.GetNumHeavyAtoms()<max_heavy_atoms
        elements = all([atom.GetAtomicNum() in element_list for atom in mol.GetAtoms()])
        if num_heavy and elements:
            return True
        else:
            return False

def write_smiles_to_file(smiles_list, fname):
    """Write a list of SMILES to a file."""
    with open(fname, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")

if __name__ == "__main__":
    smiles_list = canonicalize_smiles_from_file("data/chembl_smiles.txt")
    write_smiles_to_file(smiles_list= smiles_list, fname= "data/canonical_smiles.txt")
