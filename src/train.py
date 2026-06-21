import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from model import TCRBindingGNN

def load_graphs(graph_dir):
    files = os.listdir(graph_dir)
    feature_files = [f for f in files if f.endswith("_features.npy")]
    
    graphs = []
    for ff in feature_files:
        name = ff.replace("_features.npy", "")
        features = np.load(os.path.join(graph_dir, ff))
        edges = np.load(os.path.join(graph_dir, f"{name}_edges.npy"))
        
        x = torch.tensor(features, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long)
        y = torch.tensor([1.0], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(data)
    
    print(f"Loaded {len(graphs)} graphs")
    return graphs


def train():
    graph_dir = "data/graphs"
    graphs = load_graphs(graph_dir)
    
    negatives = []
    for g in graphs:
        neg = Data(
            x=g.x[torch.randperm(g.x.size(0))],
            edge_index=g.edge_index,
            y=torch.tensor([0.0], dtype=torch.float)
        )
        negatives.append(neg)
    
    all_graphs = graphs + negatives
    train_data, test_data = train_test_split(all_graphs, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=2)
    
    model = TCRBindingGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    loss_history = []
    auroc_history = []
    
    print("Training...")
    for epoch in range(1, 31):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch).squeeze()
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        # evaluate AUROC every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for batch in test_loader:
                    out = torch.sigmoid(model(batch.x, batch.edge_index, batch.batch).squeeze())
                    preds.extend(out.tolist())
                    labels.extend(batch.y.tolist())
            auc = roc_auc_score(labels, preds)
            auroc_history.append((epoch, auc))
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | AUROC: {auc:.4f}")
    
    # final evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = torch.sigmoid(model(batch.x, batch.edge_index, batch.batch).squeeze())
            all_preds.extend(out.tolist())
            all_labels.extend(batch.y.tolist())
    
    auc = roc_auc_score(all_labels, all_preds)
    print(f"\nFinal Test AUROC: {auc:.4f}")
    
    # confusion matrix
    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    cm = confusion_matrix(all_labels, binary_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-binding", "Binding"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: results/confusion_matrix.png")
    
    # loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 31), loss_history, color="steelblue", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig("results/loss_curve.png", dpi=150)
    plt.close()
    print("Saved: results/loss_curve.png")
    
    # AUROC curve
    epochs_tracked = [e for e, _ in auroc_history]
    aurocs = [a for _, a in auroc_history]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_tracked, aurocs, color="green", linewidth=2, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    plt.title("AUROC over Training")
    plt.tight_layout()
    plt.savefig("results/auroc_curve.png", dpi=150)
    plt.close()
    print("Saved: results/auroc_curve.png")
    
    torch.save(model.state_dict(), "results/model.pt")
    print("Model saved to results/model.pt")


if __name__ == "__main__":
    train()