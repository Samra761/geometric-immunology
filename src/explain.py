import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from model import TCRBindingGNN

def load_one_graph(graph_dir, name):
    features = np.load(os.path.join(graph_dir, f"{name}_features.npy"))
    edges = np.load(os.path.join(graph_dir, f"{name}_edges.npy"))
    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def run_explainer():
    model = TCRBindingGNN()
    model.load_state_dict(torch.load("results/model.pt", weights_only=True))
    model.eval()

    graph_dir = "data/graphs"
    name = "9ZCL" # example PDB name
    data = load_one_graph(graph_dir, name)

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="graph",
            return_type="raw",
        )
    )

    batch = torch.zeros(data.x.size(0), dtype=torch.long)
    explanation = explainer(data.x, data.edge_index, batch=batch)

    node_importance = explanation.node_mask.sum(dim=1).detach().numpy()

    print(f"Top 10 most important nodes (atoms):")
    top10 = np.argsort(node_importance)[::-1][:10]
    for i, idx in enumerate(top10):
        print(f"  {i+1}. Atom {idx} | Importance: {node_importance[idx]:.4f}")

    # plot
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(node_importance)), node_importance, color="steelblue")
    plt.xlabel("Atom Index")
    plt.ylabel("Importance Score")
    plt.title(f"Contact Residue Importance Map - {name}")
    plt.tight_layout()
    plt.savefig(f"results/{name}_importance.png", dpi=150)
    print(f"\nPlot saved to results/{name}_importance.png")

if __name__ == "__main__":
    run_explainer()