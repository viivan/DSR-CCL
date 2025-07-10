import torch.nn as nn
class PredictLayer(nn.Module):
    def __init__(self, emb_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out