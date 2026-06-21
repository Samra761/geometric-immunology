import os
import numpy as np
from scipy.spatial import KDTree

def compute_features(atoms):
    if len(atoms) == 0:
        return np.array([])
    
    tree = KDTree(atoms)
    features = []
    
    for i, atom in enumerate(atoms):
        neighbors = tree.query_ball_point(atom, r=8.0)
        neighbors = [n for n in neighbors if n != i]
        
        if len(neighbors) == 0:
            features.append([0, 0, 0, 0, 0])
            continue
        
        neighbor_coords = atoms[neighbors]
        density = len(neighbors)
        displacement = atom - neighbor_coords.mean(axis=0)
        centroid = atoms.mean(axis=0)
        dist_to_centroid = np.linalg.norm(atom - centroid)
        
        features.append([
            density,
            displacement[0],
            displacement[1],
            displacement[2],
            dist_to_centroid
        ])
    
    return np.array(features)


def build_graph(atoms, cutoff=6.0):
    tree = KDTree(atoms)
    pairs = tree.query_pairs(r=cutoff)
    edges = np.array(list(pairs)).T
    return edges


def process_all(processed_dir, graph_dir):
    os.makedirs(graph_dir, exist_ok=True)
    
    files = os.listdir(processed_dir)
    peptide_files = [f for f in files if f.endswith("_peptide.npy")]
    
    print(f"Found {len(peptide_files)} structures to process")
    
    for pf in peptide_files:
        name = pf.replace("_peptide.npy", "")
        peptide = np.load(os.path.join(processed_dir, pf))
        mhc_path = os.path.join(processed_dir, f"{name}_mhc.npy")
        mhc = np.load(mhc_path)
        
        print(f"Processing {name}...")
        
        if len(peptide) == 0 or len(mhc) == 0:
            print(f"  SKIPPED - empty array")
            continue
        
        all_atoms = np.vstack([peptide, mhc])
        peptide_label = np.zeros(len(all_atoms))
        peptide_label[:len(peptide)] = 1
        
        features = compute_features(all_atoms)
        edges = build_graph(all_atoms)
        
        np.save(os.path.join(graph_dir, f"{name}_features.npy"), features)
        np.save(os.path.join(graph_dir, f"{name}_edges.npy"), edges)
        np.save(os.path.join(graph_dir, f"{name}_labels.npy"), peptide_label)
        
        print(f"  Nodes: {len(all_atoms)}, Edges: {edges.shape[1] if edges.ndim==2 else 0}")
    
    print("\nDone!")


process_all("data/processed", "data/graphs")