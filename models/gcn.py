import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        adj_norm = adj + torch.eye(adj.shape[0])
        deg = torch.diag(torch.sum(adj_norm, dim=1)).pow(-0.5)
        adj_norm = deg @ adj_norm @ deg
        return self.linear(adj_norm @ x)
    
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, adj):
        x = self.activation(self.gcn1(x, adj))
        x = self.softmax(self.gcn2(x, adj))
        return x