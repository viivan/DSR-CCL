import torch
import torch.nn as nn
import numpy as np
from HyperConv import HyperConv
from HyperConv import HyperConv2
from PredictLayer import PredictLayer
from Attention import Attention
import torch.nn.functional as F

class MBSRHGCN(nn.Module):
    def __init__(self, num_users, num_services, num_mashups, emb_dim, layers, drop_ratio, adj, D, A, service_compose, device):
        super(MBSRHGCN, self).__init__()
        self.num_users = num_users
        self.num_services = num_services
        self.num_mashups = num_mashups
        self.emb_dim = emb_dim
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.service_embedding = nn.Embedding(num_services, self.emb_dim)
        self.layers = layers
        self.drop_ratio = drop_ratio
        self.adj = adj
        self.D = D
        self.A = A
        self.service_compose = service_compose
        self.mashup_embedding = nn.Embedding(num_mashups, self.emb_dim)
        self.hyper_graph = HyperConv(self.layers)
        self.mashup_graph = HyperConv2(self.layers)
        self.attention = Attention(2 * self.emb_dim, self.drop_ratio)
        self.predict = PredictLayer(3 * self.emb_dim, self.drop_ratio)
        self.device = device
        print(f"服务组合数量{num_mashups}")
        print(f"服务数量{num_services}")
        print(f"用户数量{num_users}")
        self.gate = nn.Sequential(nn.Linear(2 * self.emb_dim, self.emb_dim), nn.Sigmoid())

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.service_embedding.weight)
        nn.init.xavier_uniform_(self.mashup_embedding.weight)


    def forward(self, mashup_inputs, user_inputs, service_inputs):
        if (mashup_inputs is not None) and (user_inputs is None):
            # 按列拼接用户和service的向量
            ui_embedding = torch.cat((self.user_embedding.weight, self.service_embedding.weight), dim=0)
            ui_embedding = self.hyper_graph(self.adj, ui_embedding)
            ui_embedding = torch.tensor(ui_embedding)
            user_embedding, service_embedding = torch.split(ui_embedding, [self.num_users, self.num_services], dim=0)
            service_emb = service_embedding[service_inputs]

            mashup_embedding = self.mashup_graph(self.mashup_embedding.weight, self.D, self.A)

            member = []
            max_len = 0
            bsz = mashup_inputs.shape[0]
            member_masked = []
            for i in range(bsz):
                member.append(np.array(self.service_compose[mashup_inputs[i].item()]))
                max_len = max(max_len, len(self.service_compose[mashup_inputs[i].item()]))
            mask = np.zeros((bsz, max_len))
            for i, service in enumerate(member):
                cur_len = service.shape[0]
                member_masked.append(np.append(service, np.zeros(max_len - cur_len)))
                mask[i, cur_len:] = 1.0
            member_masked = torch.LongTensor(member_masked).to(self.device)
            mask = torch.Tensor(mask).to(self.device)

            member_emb = user_embedding[member_masked]
         
            service_emb_attn = service_emb.unsqueeze(1).expand(bsz, max_len, -1)
            at_emb = torch.cat((member_emb, service_emb_attn), dim=2)
            at_wt = self.attention(at_emb, mask)
            g_emb_with_attention = torch.matmul(at_wt.unsqueeze(1), member_emb).squeeze()
            g_emb_pure = mashup_embedding[mashup_inputs]
            mashup_emb = g_emb_with_attention + g_emb_pure
            element_emb = torch.mul(mashup_emb, service_emb)
            new_emb = torch.cat((element_emb, mashup_emb, service_emb), dim=1)
            y = torch.sigmoid(self.predict(new_emb))
            return y

        else:
            user_emb = self.user_embedding(user_inputs)
            service_emb = self.service_embedding(service_inputs)
            element_emb = torch.mul(user_emb, service_emb)
            new_emb = torch.cat((element_emb, user_emb, service_emb), dim=1)
            y = torch.sigmoid(self.predict(new_emb))
            return y



def loss(self, pos, aug):

    pos = pos[:, 0, :]
    aug = aug[:, 0, :]

    pos = F.normalize(pos, p=2, dim=1)
    aug = F.normalize(aug, p=2, dim=1)
    pos_score = torch.sum(pos * aug, dim=1)
    ttl_score = torch.matmul(pos, aug.permute(1, 0))
    pos_score = torch.exp(pos_score / self.c_temp)
    ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1)
    c_loss = - torch.mean(torch.log(pos_score / ttl_score))
    return c_loss



