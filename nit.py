import gzip
import shutil

def compress_txt(input_file, output_file=None):
    if output_file is None:
        output_file = input_file + '.gz'
    with open(input_file, 'rb') as f_in:
        with gzip.open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Compressed file saved as: {output_file}")

compress_txt("/Users/madukacharles/Documents/GitHub/Chem_Fuser/data/chembl_smiles.txt")