import torch
import datetime
import random
import torch.optim as optim
import numpy as np
from config import Config
from util import Helper
from HyperStruct import HyperStruc
from models.mbsr_model import MBSRHGCN
from time import time
# 使用tqdm工具可视化训练的进度条
from tqdm import tqdm
import warnings


# pytorch版本升级至2.1.2后，会提示兼容性问题，进行提示掩藏
warnings.filterwarnings("ignore")


# 训练MBSR-HGCN模型，参数从config文件中读取
# 参数解释
def train(model, train_loader, epoch_id, config, device, type_m):
    learning_rate = config.lr
    lr = learning_rate[0]
    if epoch_id >= 20 and epoch_id < 40:
        lr = learning_rate[0]
    elif epoch_id >= 40:
        lr = learning_rate[1]
    optimizer = optim.RMSprop(model.parameters(), lr)
    total_loss = []

    iterator = tqdm(enumerate(train_loader), desc=f"Epoch: {epoch_id}", total=len(train_loader))
    for batch_id, (u, s) in iterator:
        pos_service = s[:, 0].to(device)
        neg_service = s[:, 1].to(device)
        user = torch.LongTensor(u).to(device)
        if type_m == 'user':
            pos_predict = model(None, user, pos_service)
            neg_predict = model(None, user, neg_service)
        elif type_m == 'mashup':
            pos_predict = model(user, None, pos_service)
            neg_predict = model(user, None, neg_service)
        model.zero_grad()
        loss = torch.mean((pos_predict - neg_predict - 1) ** 2)
        total_loss.append(loss)
        loss.backward()
        optimizer.step()
    print('Epoch %d, %s loss is [%.4f]' % (epoch_id, type_m, torch.mean(torch.stack(total_loss))))


def evaluation(model, helper, testRatings, testNegative, device, K_list, type_m):
    model.eval()
    hits, ndcgs, mrr ,h,ils= helper.evaluate_model(model, testRatings, testNegative, device, K_list, type_m)
    return hits, ndcgs, mrr,h,ils




if __name__ == '__main__':
    config = Config()
    helper = Helper()

#   设置随机数seed，使实验结果可重复
    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True

    dataset = HyperStruc(config.user_dataset, config.mashup_dataset, config.user_in_mashup_path, config.num_negatives)

    device_id = "cuda:" + str(config.gpu_id)
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")

    num_users, num_services, num_mashups = dataset.num_users, dataset.num_services, dataset.num_mashups
    service_compose = dataset.service_compose

    # D和A都是矩阵
    adj, D, A = dataset.adj, dataset.D, dataset.A
    D = torch.Tensor(D).to(device)
    A = torch.Tensor(A).to(device)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices).to(device)
    v = torch.FloatTensor(values).to(device)
    shape = adj.shape
    adj = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)
    model = MBSRHGCN(num_users, num_services, num_mashups, config.d, config.layers, config.drop_ratio, adj, D, A,
                     service_compose, device)
    model = model.to(device)
    current_date = datetime.date.today()
    out = open("./result/dataRecord"+str(current_date)+".txt", "w")
    for epoch in range(config.epoch):
        model.train()
        t1 = time()
        print('训练行为感知卷积通道')
        train(model, dataset.get_user_dataloader(config.batch_size), epoch, config, device, 'user')
        t2 = time()
        print("训练服务组合卷积通道")
        train(model, dataset.get_mashup_dataloader(config.batch_size), epoch, config, device, 'mashup')

        # 每次训练之前先保存
        torch.save(model, './result/mbsr-model.pkl')
        # 加载训练好的参数
        # model = torch.load('./result/model.pkl')
        if epoch == 0 or epoch % 10 == 0:
            hr, ndcg, mrr,h,ils = evaluation(model, helper, dataset.mashup_testRatings, dataset.mashup_testNegatives, device,
                                       config.topK, 'mashup')

            out.write(
                'MBSR-NB Epoch %d [%.1f s]:  HR@10 = %.4f, HR@20 = %.4f; '
                'NDCG@10 = %.4f, NDCG@20 = %.4f ;MRR = %.4f ,H = %.4f,ILS = %.4f   [%.1f s]\n' % (
                    epoch, time() - t1, hr[0] / 2, hr[1] / 2, ndcg[0] / 2, ndcg[1] / 2, mrr[0] / 2, h[0]/2,ils[0]/2,time() - t2))
            for i in range(3):
                u_hr, u_ndcg, u_mrr,u_h,u_ils = evaluation(model, helper, dataset.user_testRatings, dataset.user_testNegatives,
                                                 device, config.topK, 'user')
            out.write(
                'MBSR-NC Epoch %d [%.1f s]: HR@10 = %.4f, HR@20 = %.4f; '
                'NDCG@10 = %.4f, NDCG@20 = %.4f; MRR = %.4f, H = %.4f, ILS = %.4f [%.1f s]\n' % (
                    epoch, time() - t1, u_hr[0] / 2, u_hr[1] / 2, u_ndcg[0] / 2, u_ndcg[1] / 2, u_mrr[0] / 2,u_h[0]/2,u_ils[0]/2,
                    time() - t2))
            out.write(
                'MBSR-HGCN Epoch %d [%.1f s]:  HR@10 = %.4f, HR@20 = %.4f; '
                'NDCG@10 = %.4f, NDCG@20 = %.4f; MRR = %.4f , H = %.4f, ILS = %.4f [%.1f s]\n' % (
                    epoch, time() - t1, (hr[0] + u_hr[0]) / 2, (hr[1] + u_hr[1]) / 2, (ndcg[0] + u_ndcg[0]) / 2,
                    (ndcg[1] + u_ndcg[1]) / 2, (mrr[0] + u_mrr[0]) / 2,(h[0]+u_h[0])/4,(ils[0]+u_ils[0])/2, time() - t2))
            out.write("-" * 50 + "\n")
            out.flush()
    print("训练完成")
