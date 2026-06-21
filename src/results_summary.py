import os
import numpy as np

def summarize(graph_dir):
    files = os.listdir(graph_dir)
    feature_files = sorted([f for f in files if f.endswith("_features.npy")])
    
    print("=" * 60)
    print(f"{'Structure':<10} {'Nodes':<10} {'Edges':<12} {'Top Atom':<12} {'Max Score'}")
    print("=" * 60)
    
    for ff in feature_files:
        name = ff.replace("_features.npy", "")
        features = np.load(os.path.join(graph_dir, ff))
        edges = np.load(os.path.join(graph_dir, f"{name}_edges.npy"))
        
        # importance proxy: sum of absolute feature values per node
        importance = np.abs(features).sum(axis=1)
        top_atom = np.argmax(importance)
        max_score = importance[top_atom]
        num_edges = edges.shape[1] if edges.ndim == 2 else 0
        
        print(f"{name:<10} {len(features):<10} {num_edges:<12} {top_atom:<12} {max_score:.4f}")
    
    print("=" * 60)
    print(f"\nTotal structures analyzed: {len(feature_files)}")

if __name__ == "__main__":
    summarize("data/graphs")