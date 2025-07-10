import math
import os
import random
import numpy as np
import scipy.sparse as sp
import yaml

import torch
from torch.utils.data import Dataset, DataLoader

class Helper(object):

    def __init__(self):
        self.timber = True
    def evaluate_model(self, model, testRatings, testNegatives, device, K_list, type_m):

        hits, ndcgs, mrrs,h,ils= [], [], [],[],[]
        user_test = []
        item_test = []
        # 遍历测试样本集合
        for idx in range(len(testRatings)):
            rating = testRatings[idx]
            # 真实的服务
            items = [rating[1]]
            #增加 一批负样本
            items.extend(testNegatives[idx])
            item_test.append(items)
            user = np.full(len(items), rating[0])
            user_test.append(user)
        #     将测试集样本加载至device
        users_var = torch.LongTensor(user_test).to(device)

        items_var = torch.LongTensor(item_test).to(device)

        bsz = len(testRatings)
        item_len = len(testNegatives[0]) + 1
        # 将预测结果users_var和items_var展开为列表
        users_var = users_var.view(-1)
        items_var = items_var.view(-1)
        if type_m == 'mashup':
            predictions = model(users_var, None, items_var)
        elif type_m == 'user':
            predictions = model(None, users_var, items_var)
        # 改变tensor的形状
        predictions = torch.reshape(predictions, (bsz, item_len))


        #将预测结果转化为数组np
        pred_score =  predictions.data.cpu().numpy()
        #输出服务推荐结果列表，内容为API服务的序号
        # 从小到达排序，所以要+负号，让推荐分数最大的API变为最小，axis=1代表按行排序
        pred_rank = np.argsort(pred_score *-1, axis=1)
        for k in K_list:
            hits.append(getHR(pred_rank, k//2))
            ndcgs.append(getNDCG(pred_rank, k//2))
            mrrs.append(getMRR(pred_rank, k//2))
            h.append(getH(pred_rank, k//2))
            ils.append(getILS(pred_rank,k//2))
        return hits, ndcgs, mrrs,h,ils

def getHR(pred_rank, k):
    pred_rank_k = pred_rank[:, :k]
    # 统计非0元素
    hit = np.count_nonzero(pred_rank_k == 0)
    hit = hit / pred_rank.shape[0]
    return hit

def getNDCG(pred_rank, k):
    ndcgs = np.zeros(pred_rank.shape[0])
    for user in range(pred_rank.shape[0]):
        for j in range(k):
            if pred_rank[user][j] == 0:
                ndcgs[user] = math.log(2) / math.log(j + 2)
    ndcg = np.mean(ndcgs)
    return ndcg

def getMRR(pred_rank,k):
    pred_rank_k = pred_rank[:, :k]
    mrr = 0
    for top_k_list in pred_rank_k:
        for j in range(0,k):
            if(top_k_list[j]!=0):
                mrr +=1/(j+1)
    mrr = mrr /(pred_rank.shape[0]*k)

    return mrr

def getH(pred_rank,k):
    pred_rank_k = pred_rank[:, :k]
    same = 0
    count = 0
    for u in pred_rank_k:
        for v in pred_rank_k:
            same += np.sum(u == v)
            count += len(u)
    h = 1- same/count
    return h
def getILS(pred_rank,k):
    pred_rank_k = pred_rank[:, :k]
    # 取前20个用户，加快计算
    pred_rank_k = pred_rank[:20]
    cos = 0
    num = 0
    for api_a in pred_rank_k:
        for api_b in pred_rank_k:
            cos += api_a.dot(api_b) / (np.linalg.norm(api_a) * np.linalg.norm(api_b))
            num += 1
    ils = cos/num
    return ils


    arr = pred_rank.flatten()
    count = len(arr)
    same = len(np.unique(arr))
    ils = same/count
    print(ils)
    return ils

def userItemTrain():
    file = open("./serviceDataSet/programmableWeb/recommend_user_movie_id_train_1.txt")
    oFile = open("./serviceDataSet/data/userRatingTrain.txt")
    apiSet =  set()
    for oline in oFile:
        arr = oline.split()
        apiSet.add(arr[1])
    outFile = open("./serviceDataSet/programmableWeb/userRatingTrain.txt","w")
    for line in file:
        arr = line.split()
        user = arr[0]
        arr = arr[1:]
        for i in range(len(arr)):
            if(arr[i] in apiSet):
                outFile.write(f"{user} {arr[i]}\n")

def userItemTest():
    file = open("./serviceDataSet/programmableWeb/recommend_user_movie_id_test_1.txt")
    outFile1 = open("./serviceDataSet/programmableWeb/userRatingTest.txt","w")
    outFile2 = open("./serviceDataSet/programmableWeb/userRatingNegative.txt","w")
    oFile = open("./serviceDataSet/data/userRatingTrain.txt")
    apiSet =  set()
    for oline in oFile:
        arr = oline.split()
        apiSet.add(arr[1])
    for line in file:
        arr = line.split()
        # 挑一对关系作为测试集足矣负样本需要100个
        user = arr[0]
        testAPI = arr[1]
        outFile1.write(f"{user} {arr[1]}\n")
        outStr =f"({user},{testAPI}) "
        for i in range(100):
            outStr+=str(random.choice(list(apiSet)))+" "
        outFile2.write(outStr+"\n")
def mashupHander():
    m3 = open("./serviceDataSet/data/recommend_mashup_api_id_more_than_3.txt")
    out = open("./serviceDataSet/data/3.txt","w")
    for line in m3:
        arr = line.split()
        outStr =str(arr[0])+" "
        arr = line.split()[1:]
        for i in range(len(arr)):
            outStr+=str(arr[i])+","
        length = len(outStr)
        outStr = outStr[0:length-1]+"\n"
        out.write(outStr)
# mashupHander()
def singleAPI():
    m3 = open("./serviceDataSet/data/recommend_mashup_api_id_more_than_3.txt")
    out = open("./serviceDataSet/data/3.txt","w")
    for line in m3:
        arr = line.split()
        outStr =str(arr[0])+" "
        arr = line.split()[1:]
        for i in range(len(arr)):
            outStr+=str(arr[i])+","
        length = len(outStr)
        outStr = outStr[0:length-1]+"\n"
        out.write(outStr)


def getUserService():
    userTest = open("serviceDataSet/programmableWeb/userRatingTrain.txt", "a")
    for i in range(0,15000):
        # 一名用户交互API的次数
        t = random.randint(2,4)
        #交互行为
        b = random.randint(0,4)
        for j in range(0,t):
            apiIndex = random.randint(0,12728)
            userTest.write(f"{i} {apiIndex} {b}\n")

def uiNegative():
    userTest = open("serviceDataSet/programmableWeb/userRatingTrain.txt")
    uiN = open("serviceDataSet/programmableWeb/userRatingNegative.txt", "a")
    # 读取一行
    for line in userTest:
        lineArray = line.split();
        left=str(lineArray[0])
        right=str(lineArray[1])
        outStr = "("+f"{left}"+","+f"{right}"+") "
        for i in range(0,30):
            temp = random.randint(0,12729)
            if(temp!=line[1]):
                outStr+=f"{temp} "
        outStr += "\n"
        uiN.write(outStr)

def getServiceData():
    i = 0
    apiList = []
    apiOriginal = open("./serviceDataSet/programmableWeb/apis_original.txt")
    api = open("./serviceDataSet/programmableWeb/apis.txt", "a")
    for line in apiOriginal:
        lineArray = line.split(",")
        # print(lineArray)
        # print(f"{i}  "+lineArray[0])
        # 1.加载内存，下文使用
        apiList.append((i, lineArray[0].replace(" ", "")))
        # 2.持久化 写入文件用于训练
        api.write(f"{i}:" + lineArray[0].replace(" ", "") + "\n")
        i = i + 1
    mashup = open("serviceDataSet/programmableWeb/serviceCompose.txt", "a")
    j = 0
    mashupOriginal = open("./serviceDataSet/programmableWeb/mashups_original.txt")
    for line in mashupOriginal:
        lineArray = line.split(",")
        # print(lineArray)
        apiMembers = lineArray[1].split("@@@")
        outAPI = ""
        for am in range(0, len(apiMembers)):
            ams = apiMembers[am]
            ams = ams.replace(" ", "")
            for al in apiList:
                if ams == al[1]:
                    outAPI += str(al[0])
                    outAPI += ","
        outAPI = outAPI[0:len(outAPI) - 1]
        if(outAPI!=""):
            mashup.write(f"{j} " + outAPI + "\n")
            j = j + 1
def getPopData():
    testRatings = open("./serviceDataSet/data/userRatingTest.txt")
    out = open("./serviceDataSet/data/merge.txt",'a')
    i = 0
    for line  in testRatings:
        j = 0
        NegtiveRating = open("./serviceDataSet/data/userRatingNegative.txt")
        for nLine in NegtiveRating:
            print(i, j)
            items = [line.split()[1]]
            user = line.split()[0]
            if(i==j):
                items.extend( nLine.split()[1:])
                items =list( map(int,items))
                out.write(user)
                out.write(" ")
                for l in items:
                    out.write(str(l))
                    out.write(" ")
                out.write("\n")
            j=j+1
        i = i + 1

        # 真实的服务

def print_statistics(X, string):
    print('>'*10 + string + '>'*10 )
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))


class BundleTrainDataset(Dataset):
    def __init__(self, conf, u_b_pairs, u_b_graph, num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, neg_sample=1):
        self.conf = conf
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.num_bundles = num_bundles
        self.neg_sample = neg_sample

        self.u_b_for_neg_sample = u_b_for_neg_sample
        self.b_b_for_neg_sample = b_b_for_neg_sample


    def __getitem__(self, index):
        conf = self.conf
        user_b, pos_bundle = self.u_b_pairs[index]
        all_bundles = [pos_bundle]

        while True:
            i = np.random.randint(self.num_bundles)
            if self.u_b_graph[user_b, i] == 0 and not i in all_bundles:
                all_bundles.append(i)
                if len(all_bundles) == self.neg_sample+1:
                    break

        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)


    def __len__(self):
        return len(self.u_b_pairs)


class BundleTestDataset(Dataset):
    def __init__(self, u_b_pairs, u_b_graph, u_b_graph_train, num_users, num_bundles):
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.train_mask_u_b = u_b_graph_train

        self.num_users = num_users
        self.num_bundles = num_bundles

        self.users = torch.arange(num_users, dtype=torch.long).unsqueeze(dim=1)
        self.bundles = torch.arange(num_bundles, dtype=torch.long)


    def __getitem__(self, index):
        u_b_grd = torch.from_numpy(self.u_b_graph[index].toarray()).squeeze()
        u_b_mask = torch.from_numpy(self.train_mask_u_b[index].toarray()).squeeze()

        return index, u_b_grd, u_b_mask


    def __len__(self):
        return self.u_b_graph.shape[0]


class Datasets():
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']

        self.num_users, self.num_bundles, self.num_items = self.get_data_size()

        b_i_graph = self.get_bi()
        u_i_pairs, u_i_graph = self.get_ui()

        u_b_pairs_train, u_b_graph_train = self.get_ub("train")
        u_b_pairs_val, u_b_graph_val = self.get_ub("tune")
        u_b_pairs_test, u_b_graph_test = self.get_ub("test")

        u_b_for_neg_sample, b_b_for_neg_sample = None, None

        self.bundle_train_data = BundleTrainDataset(conf, u_b_pairs_train, u_b_graph_train, self.num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, conf["neg_num"])
        self.bundle_val_data = BundleTestDataset(u_b_pairs_val, u_b_graph_val, u_b_graph_train, self.num_users, self.num_bundles)
        self.bundle_test_data = BundleTestDataset(u_b_pairs_test, u_b_graph_test, u_b_graph_train, self.num_users, self.num_bundles)

        self.graphs = [u_b_graph_train, u_i_graph, b_i_graph]

        self.train_loader = DataLoader(self.bundle_train_data, batch_size=batch_size_train, shuffle=True, num_workers=0, drop_last=True)
        self.val_loader = DataLoader(self.bundle_val_data, batch_size=batch_size_test, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.bundle_test_data, batch_size=batch_size_test, shuffle=False, num_workers=0)


    def get_data_size(self):
        name = self.name
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]


    def get_aux_graph(self, u_i_graph, b_i_graph, conf):
        u_b_from_i = u_i_graph @ b_i_graph.T
        u_b_from_i = u_b_from_i.todense()
        bn1_window = [int(i*self.num_bundles) for i in conf['hard_window']]
        u_b_for_neg_sample = np.argsort(u_b_from_i, axis=1)[:, bn1_window[0]:bn1_window[1]]

        b_b_from_i = b_i_graph @ b_i_graph.T
        b_b_from_i = b_b_from_i.todense()
        bn2_window = [int(i*self.num_bundles) for i in conf['hard_window']]
        b_b_for_neg_sample = np.argsort(b_b_from_i, axis=1)[:, bn2_window[0]:bn2_window[1]]

        return u_b_for_neg_sample, b_b_for_neg_sample


    def get_bi(self):
        with open(os.path.join(self.path, self.name, 'bundle_item.txt'), 'r') as f:
            b_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split(' ')), f.readlines()))

        indice = np.array(b_i_pairs, dtype=np.int32)
        values = np.ones(len(b_i_pairs), dtype=np.float32)
        b_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_bundles, self.num_items)).tocsr()

        print_statistics(b_i_graph, 'B-I statistics')

        return b_i_graph


    def get_ui(self):
        with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
            u_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split(' ')), f.readlines()))

        indice = np.array(u_i_pairs, dtype=np.int32)
        values = np.ones(len(u_i_pairs), dtype=np.float32)
        u_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items)).tocsr()

        print_statistics(u_i_graph, 'U-I statistics')

        return u_i_pairs, u_i_graph


    def get_ub(self, task):
        with open(os.path.join(self.path, self.name, 'user_bundle_{}.txt'.format(task)), 'r') as f:
            u_b_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split(' ')), f.readlines()))

        indice = np.array(u_b_pairs, dtype=np.int32)
        values = np.ones(len(u_b_pairs), dtype=np.float32)
        u_b_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()

        print_statistics(u_b_graph, "U-B statistics in %s" %(task))

        return u_b_pairs, u_b_graph

dsr_conf = yaml.safe_load(open("config.yaml"))
dsr_conf = dsr_conf["serviceData"]
laplace = dsr_conf["laplace"]
def laplace_transform(graph):
    c_s = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    r_s = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    graph = r_s @ graph @ c_s

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
    return graph


# 边舍弃数据增强
def edge_drop(edge, ratio):
    mask = np.random.choice([0, 1], size=(len(edge),), p=[ratio, 1-ratio])
    return mask*edge


# 节点舍弃数据增强
def node_drop(node,ratio):
    mask = np.random.choice([0, 1], size=(len(node),), p=[2*ratio, 1 - 2*ratio])
    return mask *node
