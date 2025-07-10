import torch
import torch.nn as nn
import numpy as np

# 使用注意力机制对超图信号进行度量
class Attention(nn.Module):
    def __init__(self, emb_dim, drop_ratio=0):
        super(Attention, self).__init__()
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, int(emb_dim / 2)),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(int(emb_dim / 2), 1)
        )

    def forward(self, x, mask):
        bsz = x.shape[0]
        out = self.linear(x)
        out = out.view(bsz, -1) # [bsz, max_len]
        out.masked_fill_(mask.bool(), -np.inf)
        weight = torch.softmax(out, dim=1)
        return weight
