import torch
import torch.nn as nn
import numpy as np

# 切比雪夫优化后的超图卷积算子
class HyperConv(nn.Module):
    def __init__(self, layers):
        super(HyperConv, self).__init__()
        self.layers = layers
    # 输入为： adj:邻接矩阵  emb:嵌入向量
    def forward(self, adj, emb):
        emb_all = emb
        final = [emb_all]
        for i in range(self.layers):
            emb_all = torch.sparse.mm(adj, emb_all)
            final.append(emb_all)
        final_emb = np.sum(final, 0)
        return final_emb

# 另一种超图卷积方法
class HyperConv2(nn.Module):
    def __init__(self, layers):
        super(HyperConv2, self).__init__()
        self.layers = layers

    def forward(self, embedding, D, A):
        DA = torch.mm(D, A).float()
        mashup_emb = embedding
        final = [mashup_emb]
        for i in range(self.layers):
            mashup_emb = torch.mm(DA, mashup_emb)
            final.append(mashup_emb)
        final_emb = np.sum(final, 0)
        return final_emb
