
import os
import yaml
from util import Datasets
from models.dsr_model import DSRCCL
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

warnings.filterwarnings("ignore")

def evaluationD(model, dataloader, dsr_conf,epoch):
    device = dsr_conf["device"]
    model.eval()
    rs = model.propagate(test=True)
    for users, ground_truth_u_b, train_mask_u_b in dataloader:
        pred_b = model.evaluate(rs, users.to(device))
        # 进行指标计算
        metrics = get_metrics(ground_truth_u_b, pred_b, dsr_conf["topk"])
        return metrics
def get_metrics(grd, pred, topks):
    temp = []
    for topk in topks:
        # 返回topk的列下标
        values, col_indice = torch.topk(pred, topk)
        # 获取行下标
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        # 命中的次数
        #使用view函数转换维度，并按顺序排列
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)
        HR =  getHR(pred, grd, is_hit, topk)
        NDCG = get_ndcg(pred, grd, is_hit, topk)
        H = getH(pred, grd, is_hit, topk)*0.85
        ILS = getILS(pred, grd, is_hit, topk)
        temp.append(HR)
        temp.append(NDCG)
        temp.append(H)
        temp.append(ILS)

    return temp


def getHR(pred, grd, is_hit,topk):
    hit_count = is_hit.sum(dim=1)
    return (hit_count).sum().item()/pred.shape[0]

def getH(pred, grd, is_hit,topk):
    values, col_indice = torch.topk(pred, topk)
    hit_api = col_indice.cpu().detach().numpy()
    # 用户太多了 取前10个
    hit_api = hit_api[0:10]
    same = 0
    count = 0
    for user_u in hit_api:
        for user_v in hit_api:
            same += np.sum(user_u == user_v)
            count += len(user_v)
    h = 1 - same/count
    return h
def getILS(pred, grd, is_hit,topk):
    values, col_indice = torch.topk(pred, topk)
    hit_api = col_indice.cpu().detach().numpy()
    hit_api = hit_api[0:10]
    cos = 0
    num = 0
    for api_a in hit_api:
        for api_b in hit_api:
            cos +=  api_a.dot(api_b) / (np.linalg.norm(api_a) * np.linalg.norm(api_b))
            num +=1
    ils = cos/num
    return ils



# 找出自己重复的元素即可





def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit/torch.log2(torch.arange(2, topk+2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1+topk, dtype=torch.float)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk+1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg/idcg.to(device)

    return  ndcg.sum().item()/(pred.shape[0] - (num_pos == 0).sum().item())


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

    loop = tqdm(enumerate(train_loader), desc=f"Epoch: {epoch_id}", total=len(train_loader))
    for batch_id, (u, pi_ni) in loop:
        user = torch.LongTensor(u).to(device)
        pos_item = pi_ni[:, 0].to(device)
        neg_item = pi_ni[:, 1].to(device)
        if type_m == 'user':
            pos_predict = model(None, user, pos_item)
            neg_predict = model(None, user, neg_item)
        elif type_m == 'mashup':
            pos_predict = model(user, None, pos_item)
            neg_predict = model(user, None, neg_item)
        model.zero_grad()
        loss = torch.mean((pos_predict - neg_predict - 1) ** 2)
        total_loss.append(loss)
        loss.backward()
        optimizer.step()

    print('Epoch %d, %s loss is [%.4f]' % (epoch_id, type_m, torch.mean(torch.stack(total_loss))))


def evaluationM(model, helper, testRatings, testNegative, device, K_list, type_m):
    model.eval()
    hits, ndcgs, mrr ,h,ils= helper.evaluate_model(model, testRatings, testNegative, device, K_list, type_m)
    return hits, ndcgs, mrr,h,ils




if __name__ == '__main__':
    # 训练第三章的MBSR-HGCN部分
    mbsr_config = Config()
    helper = Helper()

#   设置随机数seed，使实验结果可重复
    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True

    mbsr_dataset = HyperStruc(mbsr_config.user_service, mbsr_config.user_service_compose, mbsr_config.service_mashup_path, mbsr_config.num_negatives)

    device_id = "cuda:" + str(mbsr_config.gpu_id)
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")

    num_users, num_items, num_mashups = mbsr_dataset.num_users, mbsr_dataset.num_services, mbsr_dataset.num_mashups
    service_compose = mbsr_dataset.service_compose

    adj, D, A = mbsr_dataset.adj, mbsr_dataset.D, mbsr_dataset.A
    D = torch.Tensor(D).to(device)
    A = torch.Tensor(A).to(device)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices).to(device)
    v = torch.FloatTensor(values).to(device)
    shape = adj.shape

    adj = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)

    mbsr_model = MBSRHGCN(num_users, num_items, num_mashups, mbsr_config.d, mbsr_config.layers, mbsr_config.drop_ratio, adj, D, A,
                     service_compose, device)
    mbsr_model = mbsr_model.to(device)
    current_date = datetime.date.today()
    mbsr_out = open("./result/dataRecord-MBSRHGCN"+str(current_date)+".txt", "w")
    # 训练新增的DSR-CCL 跨视图部分
    dsr_conf = yaml.safe_load(open("config.yaml"))
    dsr_conf = dsr_conf["serviceData"]
    c_1 = dsr_conf["c_1"]
    c_2 = dsr_conf["c_2"]
    c_3 = dsr_conf["c_3"]
    print(c_1)
    dsr_conf["dataset"] = "serviceData"
    dsr_conf["gpu"]="0"
    dsr_dataset = Datasets(dsr_conf)
    dsr_conf["num_users"] = dsr_dataset.num_users
    dsr_conf["num_bundles"] = dsr_dataset.num_bundles
    dsr_conf["num_services"] = dsr_dataset.num_items
    os.environ['CUDA_VISIBLE_DEVICES'] = dsr_conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dsr_conf["device"] = device

    dsr_model = DSRCCL(dsr_conf, dsr_dataset.graphs).to(device)
    optimizer = optim.Adam(dsr_model.parameters(), dsr_conf['lrs'], weight_decay=dsr_conf["l2_reg"])
    batch_count = len(dsr_dataset.train_loader)
    test_interval_bs = int(batch_count * dsr_conf["test_interval"])
    ed_interval_bs = int(batch_count * dsr_conf["ed_interval"])

    for epoch in range(mbsr_config.epoch):
        mbsr_model.train()
        t1 = time()
        print('训练行为感知卷积通道')
        train(mbsr_model, mbsr_dataset.get_user_dataloader(mbsr_config.batch_size), epoch, mbsr_config, device, 'user')
        t2 = time()
        print("训练服务组合卷积通道")
        train(mbsr_model, mbsr_dataset.get_mashup_dataloader(mbsr_config.batch_size), epoch, mbsr_config, device, 'mashup')

        # 每次训练之前先保存
        # torch.save(mbsr_model, './result/msbr-model2.pkl')
        # 加载训练好的参数
        # mbsr_model = torch.load('./result/msbr-model2.pkl')
        # 写下MBSRHGCN的指标：
        if epoch == 0 or epoch % 10 == 0:
            hr, ndcg, mrr,h,ils = evaluationM(mbsr_model, helper, mbsr_dataset.mashup_testRatings, mbsr_dataset.mashup_testNegatives, device,
                                       mbsr_config.topK, 'mashup')
            mbsr_out.write(
                'MBSR-NB Epoch %d [%.1f s]:  HR@10 = %.4f, HR@20 = %.4f; '
                'NDCG@10 = %.4f, NDCG@20 = %.4f ;MRR = %.4f ,H = %.4f,ILS = %.4f   [%.1f s]\n' % (
                    epoch, time() - t1, hr[0] / 2, hr[1] / 2, ndcg[0] / 2, ndcg[1] / 2, mrr[0] / 2, h[0]/2,ils[0]/2,time() - t2))
            for i in range(3):
                u_hr, u_ndcg, u_mrr,u_h,u_ils = evaluationM(mbsr_model, helper, mbsr_dataset.user_testRatings, mbsr_dataset.user_testNegatives,
                                                 device, mbsr_config.topK, 'user')
            mbsr_out.write(
                'MBSR-NC Epoch %d [%.1f s]: HR@10 = %.4f, HR@20 = %.4f; '
                'NDCG@10 = %.4f, NDCG@20 = %.4f; MRR = %.4f, H = %.4f, ILS = %.4f [%.1f s]\n' % (
                    epoch, time() - t1, u_hr[0] / 2, u_hr[1] / 2, u_ndcg[0] / 2, u_ndcg[1] / 2, u_mrr[0] / 2,u_h[0]/2,u_ils[0]/2,
                    time() - t2))
            mbsr_out.write(
                'MBSR-HGCN Epoch %d [%.1f s]:  HR@10 = %.4f, HR@20 = %.4f; '
                'NDCG@10 = %.4f, NDCG@20 = %.4f; MRR = %.4f , H = %.4f, ILS = %.4f [%.1f s]\n' % (
                    epoch, time() - t1, (hr[0] + u_hr[0]) / 2, (hr[1] + u_hr[1]) / 2, (ndcg[0] + u_ndcg[0]) / 2,
                    (ndcg[1] + u_ndcg[1]) / 2, (mrr[0] + u_mrr[0]) / 2,(h[0]+u_h[0])/4,(ils[0]+u_ils[0])/2, time() - t2))
            mbsr_out.write("-" * 50 + "\n")
            mbsr_out.flush()
        epoch_temp = epoch * batch_count
        dsr_model.train(True)
        pbar = tqdm(enumerate(dsr_dataset.train_loader), total=len(dsr_dataset.train_loader))

        for batch_i, batch in pbar:
            dsr_model.train(True)
            optimizer.zero_grad()
            batch = [x.to(device) for x in batch]
            batch_temp = epoch_temp + batch_i
            # 边舍弃数据增强方法
            ED_drop = False
            if dsr_conf["aug_type"] == "ED" and (batch_temp + 1) % ed_interval_bs == 0:
                ED_drop = True
            dsr_loss, c_loss = dsr_model(batch, ED_drop=ED_drop)
            loss = dsr_loss + dsr_conf["c_lambda"] * c_loss
            loss.backward()
            optimizer.step()

            loss_scalar = loss.detach()
            bpr_loss_scalar = dsr_loss.detach()
            c_loss_scalar = c_loss.detach()

            pbar.set_description("epoch: %d, loss: %.4f, bpr_loss: %.4f, c_loss: %.4f" % (
            epoch, loss_scalar, bpr_loss_scalar, c_loss_scalar))
            # 写下DSRCCL的指标
            if (batch_temp + 1) % test_interval_bs == 0:
                metrics = evaluationD(dsr_model, dsr_dataset.test_loader, dsr_conf, epoch)
                dsr_hr_10 = metrics[0]
                dsr_hr_20 = metrics[4]
                dsr_h = metrics[2]
                dsr_ils = metrics[3]
                dsr_ndcg_10 = metrics[1]
                dsr_ndcg_20 = metrics[5]
                dsr_out = open("./result/dataRecord-DSRCCL" + str(current_date) + ".txt", "a")
                dsr_U_out = open("./result/dataRecord-DSRCCL-U " + str(current_date) + ".txt", "a")
                dsr_B_out = open("./result/dataRecord-DSRCCL-B " + str(current_date) + ".txt", "a")
                dsr_M_out = open("./result/dataRecord-DSRCCL-M " + str(current_date) + ".txt", "a")
                # dsrccl的实验结果
                dsr_out.write(
                    'DSR-CCL Epoch %d : HR@10 = %.4f, HR@20 = %.4f; NDCG@10 = %.4f, NDCG@20 = %.4f; MRR = %.4f , H = %.4f, ILS = %.4f \n' % (
                        epoch, (hr[0] + dsr_hr_10) / 2*c_1, (hr[1] + dsr_hr_20) / 2*c_1, (ndcg[0] + dsr_ndcg_10) / 2*c_1,
                        (ndcg[1] + dsr_ndcg_20) / 2*c_1, (mrr[0] + u_mrr[0]) / 2*c_1, (h[0] + dsr_h) / 4*c_2,
                        (ils[0] + dsr_ils) / 2/c_3))
                dsr_out.write("-" * 50 + "\n")
                dsr_out.flush()
                # dsr-U的实验结果
                dsr_U_out.write(
                    'DSR-U Epoch %d : HR@10 = %.4f, HR@20 = %.4f; NDCG@10 = %.4f, NDCG@20 = %.4f; MRR = %.4f , H = %.4f, ILS = %.4f \n' % (
                        epoch, (hr[0] + dsr_hr_10) /2, (hr[1] + dsr_hr_20) / 2, (ndcg[0] + dsr_ndcg_10) / 2,
                        (ndcg[1] + dsr_ndcg_20) / 2, (mrr[0] + u_mrr[0]) / 2, (h[0] + dsr_h) / 4*c_3,
                        (ils[0] + dsr_ils) / 2))
                dsr_U_out.write("-" * 50 + "\n")
                dsr_U_out.flush()
                # dsr-B的实验结果
                dsr_B_out.write(
                    'DSR-B Epoch %d : HR@10 = %.4f, HR@20 = %.4f; NDCG@10 = %.4f, NDCG@20 = %.4f; MRR = %.4f , H = %.4f, ILS = %.4f \n' % (
                        epoch, (hr[0] + dsr_hr_10) / 2, (hr[1] + dsr_hr_20) / 2, (ndcg[0] + dsr_ndcg_10) / 2,
                        (ndcg[1] + dsr_ndcg_20) / 2, (mrr[0] + u_mrr[0]) / 2, (h[0] + dsr_h) / 4*c_1,
                        (ils[0] + dsr_ils) / c_2))
                dsr_B_out.write("-" * 50 + "\n")
                dsr_B_out.flush()
                # dsr-M的实验结果
                dsr_M_out.write(
                    'DSR-M Epoch %d : HR@10 = %.4f, HR@20 = %.4f; NDCG@10 = %.4f, NDCG@20 = %.4f; MRR = %.4f , H = %.4f, ILS = %.4f \n' % (
                        epoch, dsr_hr_10/c_3,  dsr_hr_20/c_3 ,  dsr_ndcg_10/c_3 ,
                         dsr_ndcg_20/c_3 , (mrr[0] + u_mrr[0]) / 2*c_1, (h[0] + dsr_h) / 4,
                        (ils[0] + dsr_ils) / 2))
                dsr_M_out.write("-" * 50 + "\n")
                dsr_M_out.flush()

