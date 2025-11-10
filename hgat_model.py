import torch.nn as nn
import torch

class SimpleHGAT(nn.Module):
    def __init__(self, feat_dim, hidden_dim, vocab_size):
        super(SimpleHGAT, self).__init__()
        self.linear1 = nn.Linear(feat_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, edge_index=None):
        h = self.relu(self.linear1(x))
        out = self.decoder(h)
        return out
