import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_importance(processed_dir, graph_dir, name, out_dir):
    peptide = np.load(os.path.join(processed_dir, f"{name}_peptide.npy"))
    mhc = np.load(os.path.join(processed_dir, f"{name}_mhc.npy"))
    all_atoms = np.vstack([peptide, mhc])

    features = np.load(os.path.join(graph_dir, f"{name}_features.npy"))
    importance = np.abs(features).sum(axis=1)

    # subsample for speed (max 1000 points)
    idx = np.random.choice(len(all_atoms), min(1000, len(all_atoms)), replace=False)
    coords = all_atoms[idx]
    scores = importance[idx]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=scores, cmap='plasma', s=5, alpha=0.7
    )

    plt.colorbar(sc, ax=ax, label='Importance Score')
    ax.set_title(f'3D Contact Importance Map — {name}')
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{name}_3d_importance.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    structures = [
        "1AO7", "2BNR", "2CKB", "3QEQ", "4MNQ", "5TEZ",
        "6EQA", "7T2B", "9NMU", "9NMV", "9PBG", "9YW4", "9ZCL"
    ]
    os.makedirs("results", exist_ok=True)
    for name in structures:
        print(f"Processing {name}...")
        plot_3d_importance("data/processed", "data/graphs", name, "results")
    print("\nAll 3D plots saved.")