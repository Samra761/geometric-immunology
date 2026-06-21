import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class TCRBindingGNN(nn.Module):
    def __init__(self, in_channels=5, hidden=64, out_channels=1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden * 2)
        self.conv3 = GCNConv(hidden * 2, hidden)
        self.fc = nn.Linear(hidden, out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.fc(x)


if __name__ == "__main__":
    print("Model defined successfully")
    model = TCRBindingGNN()
    print(model)