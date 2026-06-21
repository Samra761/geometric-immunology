import os
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from model import TCRBindingGNN

def load_graphs(graph_dir):
    files = os.listdir(graph_dir)
    feature_files = [f for f in files if f.endswith("_features.npy")]
    
    graphs = []
    for ff in feature_files:
        name = ff.replace("_features.npy", "")
        features = np.load(os.path.join(graph_dir, ff))
        edges = np.load(os.path.join(graph_dir, f"{name}_edges.npy"))
        labels = np.load(os.path.join(graph_dir, f"{name}_labels.npy"))
        
        x = torch.tensor(features, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long)
        
        # graph label: 1 if has TCR binding peptide (all our structures do)
        y = torch.tensor([1.0], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(data)
    
    print(f"Loaded {len(graphs)} graphs")
    return graphs


def train():
    graph_dir = "data/graphs"
    graphs = load_graphs(graph_dir)
    
    # since all are positive, create negative samples by shuffling features
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
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f}")
    
    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            out = torch.sigmoid(model(batch.x, batch.edge_index, batch.batch).squeeze())
            all_preds.extend(out.tolist())
            all_labels.extend(batch.y.tolist())
    
    auc = roc_auc_score(all_labels, all_preds)
    print(f"\nTest AUROC: {auc:.4f}")
    
    torch.save(model.state_dict(), "results/model.pt")
    print("Model saved to results/model.pt")


import torch.nn as nn
if __name__ == "__main__":
    train()