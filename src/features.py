import os
import numpy as np
from scipy.spatial import KDTree
from Bio import PDB

# Kyte-Doolittle hydropathy scale
HYDROPATHY = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5,
    'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4,
    'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9,
    'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8,
    'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}

def get_hydropathy(residue_name):
    return HYDROPATHY.get(residue_name, 0.0)

def extract_atoms_with_hydropathy(pdb_file, peptide_range):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_file)
    
    coords = []
    hydro = []
    
    for model in structure:
        for chain in model:
            residues = [r for r in chain.get_residues() if PDB.is_aa(r)]
            for res in residues:
                h = get_hydropathy(res.get_resname())
                for atom in res.get_atoms():
                    coords.append(atom.get_vector().get_array())
                    hydro.append(h)
    
    return np.array(coords), np.array(hydro)

def compute_features(atoms, hydro):
    if len(atoms) == 0:
        return np.array([])
    
    tree = KDTree(atoms)
    features = []
    
    for i, atom in enumerate(atoms):
        neighbors = tree.query_ball_point(atom, r=8.0)
        neighbors = [n for n in neighbors if n != i]
        
        if len(neighbors) == 0:
            features.append([0, 0, 0, 0, 0, 0])
            continue
        
        neighbor_coords = atoms[neighbors]
        density = len(neighbors)
        displacement = atom - neighbor_coords.mean(axis=0)
        centroid = atoms.mean(axis=0)
        dist_to_centroid = np.linalg.norm(atom - centroid)
        h_score = hydro[i]
        
        features.append([
            density,
            displacement[0],
            displacement[1],
            displacement[2],
            dist_to_centroid,
            h_score
        ])
    
    return np.array(features)

def build_graph(atoms, cutoff=6.0):
    tree = KDTree(atoms)
    pairs = tree.query_pairs(r=cutoff)
    edges = np.array(list(pairs)).T
    return edges

def process_all(pdb_dir, processed_dir, graph_dir):
    os.makedirs(graph_dir, exist_ok=True)
    
    files = os.listdir(processed_dir)
    peptide_files = [f for f in files if f.endswith("_peptide.npy")]
    
    print(f"Found {len(peptide_files)} structures")
    
    for pf in peptide_files:
        name = pf.replace("_peptide.npy", "")
        pdb_path = os.path.join(pdb_dir, f"{name}.pdb")
        
        if not os.path.exists(pdb_path):
            print(f"  SKIPPED {name} - no PDB file")
            continue
        
        peptide = np.load(os.path.join(processed_dir, pf))
        mhc = np.load(os.path.join(processed_dir, f"{name}_mhc.npy"))
        
        if len(peptide) == 0 or len(mhc) == 0:
            print(f"  SKIPPED {name} - empty array")
            continue
        
        print(f"Processing {name}...")
        
        coords, hydro = extract_atoms_with_hydropathy(pdb_path, len(peptide))
        
        if len(coords) == 0:
            print(f"  SKIPPED {name} - no coords")
            continue
        
        features = compute_features(coords, hydro)
        edges = build_graph(coords)
        
        all_atoms = np.vstack([peptide, mhc])
        labels = np.zeros(len(all_atoms))
        labels[:len(peptide)] = 1
        
        np.save(os.path.join(graph_dir, f"{name}_features.npy"), features)
        np.save(os.path.join(graph_dir, f"{name}_edges.npy"), edges)
        np.save(os.path.join(graph_dir, f"{name}_labels.npy"), labels)
        
        print(f"  Nodes: {len(features)}, Edges: {edges.shape[1] if edges.ndim==2 else 0}")
    
    print("\nDone!")

process_all("data/raw_pdbs", "data/processed", "data/graphs")