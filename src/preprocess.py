print("Script started")

import os
from Bio import PDB
import numpy as np

def extract_pMHC_atoms(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_file)
    
    peptide_atoms = []
    mhc_atoms = []
    
    for model in structure:
        chains = list(model.get_chains())
        
        # count residues per chain
        chain_lengths = {}
        for chain in chains:
            residues = [r for r in chain.get_residues() if PDB.is_aa(r)]
            chain_lengths[chain.id] = len(residues)
        
        print(f"  Chain lengths: {chain_lengths}")
        
        # peptide = chain with 7-25 amino acids
        for chain in model:
            residues = [r for r in chain.get_residues() if PDB.is_aa(r)]
            n = len(residues)
            if 7 <= n <= 25:
                for atom in chain.get_atoms():
                    peptide_atoms.append(atom.get_vector().get_array())
            elif n > 25:
                for atom in chain.get_atoms():
                    mhc_atoms.append(atom.get_vector().get_array())
    
    return np.array(peptide_atoms), np.array(mhc_atoms)


def process_all(pdb_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    
    print(f"Found {len(pdb_files)} PDB files")
    
    for pdb_file in pdb_files:
        name = pdb_file.replace(".pdb", "")
        path = os.path.join(pdb_dir, pdb_file)
        print(f"\nProcessing {name}...")
        
        try:
            peptide, mhc = extract_pMHC_atoms(path)
            print(f"  Peptide atoms: {len(peptide)}, MHC atoms: {len(mhc)}")
            
            if len(peptide) > 0:
                np.save(os.path.join(out_dir, f"{name}_peptide.npy"), peptide)
                np.save(os.path.join(out_dir, f"{name}_mhc.npy"), mhc)
                print(f"  Saved!")
            else:
                print(f"  SKIPPED - no peptide detected")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print("\nDone!")


process_all("data/raw_pdbs", "data/processed")