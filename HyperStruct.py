import torch
import scipy.sparse as sp
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict




class HyperStruc(object):

    def __init__(self, user_path, mashup_path, user_in_mashup_path, num_negatives):
        self.num_negatives = num_negatives
        # user data
        # 用户-项目
        self.user_trainMatrix = self.load_rating_file_as_matrix(user_path + "Train.txt")
        # 测试
        self.user_testRatings = self.load_rating_file_as_list(user_path + "Test.txt")
        # 负样本
        self.user_testNegatives = self.load_negative_file(user_path + "Negative.txt")
        self.num_users, self.num_services = self.user_trainMatrix.shape
        # mashup data
        # 用户-bundle
        self.mashup_trainMatrix = self.load_rating_file_as_matrix(mashup_path + "Train.txt")
        # 测试
        self.mashup_testRatings = self.load_rating_file_as_list(mashup_path + "Test.txt")
        # 正样本
        self.mashup_testNegatives = self.load_negative_file(mashup_path + "Negative.txt")
        self.num_mashups = self.mashup_trainMatrix.shape[0]
        self.adj, mashup_data, self.service_compose = self.get_hyper_adj(user_in_mashup_path, mashup_path + "Train.txt")
        self.D, self.A = self.get_mashup_adj(mashup_data)

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, service = int(arr[0]), int(arr[1])
                ratingList.append([user, service])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                negatives = []
                for x in arr[1:]:
                    if(x!="\n"):
                        negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    # 邻接矩阵构建算法
    def load_rating_file_as_matrix(self, filename):
        # Get number of users and services
        num_users = 0
        num_services = 0
        with open(filename, "r") as f:
            line = f.readline()
            # 读入一行
            while line != None and line != "":
                arr = line.split(" ")
                # 数据集中,第一列是用户的id,第二列是物品id,两者都是有顺序的,取最大值就是矩阵的行列数
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_services = max(num_services, i)
                line = f.readline()
        # 利用scipy.sparse包 构建矩阵,dok_matrix说明是基于字典数据结构创建稀疏矩阵
        mat = sp.dok_matrix((num_users + 1, num_services + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                if len(arr) > 2:
                    user, service, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    if (rating > 0):
                        mat[user, service] = 1.0
                else:
                    # 不使用注意力机制,默认为1,不考虑交互行为,默认为1
                    user, service = int(arr[0]), int(arr[1])
                    mat[user, service] = 1.0
                line = f.readline()
        return mat

    # 超图构建算法
    def get_hyper_adj(self, user_in_mashup_path, mashup_train_path):
        print("*"*20+"构造超图"+"*"*20)
        g_m_d = {}
        # 服务组合超图
        with open(user_in_mashup_path, 'r') as f:
            line = f.readline().strip()
            # 读入一行
            while line != None and line != "":
                a = line.split(' ')
                # 第一列代表mashup服务组合的id
                g = int(a[0])
                # 第二列是一个用逗号分割的数组,每一个元素都是API服务的id
                g_m_d[g] = []
                for m in a[1].split(','):
                    g_m_d[g].append(int(m))
                line = f.readline().strip()
        # 创建一个字典g_i_d
        g_i_d = defaultdict(list)
        # 用户和服务组合之间的交互记录
        with open(mashup_train_path, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                if len(arr) > 2:
                    mashup, service, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    if (rating > 0):
                        # 为什么需要增加self.num_users
                        g_i_d[mashup].append(service + self.num_users)
                else:
                    mashup, service = int(arr[0]), int(arr[1])
                    g_i_d[mashup].append(service + self.num_users)
                line = f.readline()
        mashup_data = []

        for i in range(self.num_mashups):
            # 组数
            mashup_data.append(g_m_d[i] + g_i_d[i])
            # mashup_data.append(g_m_d[i])
        def _hyperGenerator_(all_mashup_data):
            # 参数解释：indptr 第一行有几个是非零的， indices 行中第几列是非零的 data存储所有数据，
            indptr, indices, data = [], [], []
            # 这个0是为了计算行中含有非零值个数添加的
            indptr.append(0)
            for j in range(len(all_mashup_data)):
                single_mashup = np.unique(np.array(all_mashup_data[j]))
                length = len(single_mashup)
                s = indptr[-1]
                indptr.append(s + length)
                for i in range(length):
                    indices.append(single_mashup[i])
                    # 超图的权值全是1
                    data.append(1)
            #         行数 服务组合数量 ；列数 API数
            print(f"矩阵的行数{self.num_mashups}",f"列数{self.num_users + self.num_services}")
            matrix = sp.csr_matrix((data, indices, indptr), shape=(self.num_mashups, self.num_users + self.num_services))
            return matrix
        H_T = _hyperGenerator_(mashup_data)
        # BH_T矩阵的转置,取负1次
        BH_T = H_T.T.multiply(1.0/(1.0 + H_T.sum(axis=1).reshape(1, -1)))
        BH_T = BH_T.T
        # 对矩阵进行转置
        H = H_T.T
        # 取他的负一次,再次转置
        DH = H.T.multiply(1.0/(1.0 + H.sum(axis=1).reshape(1, -1)))
        DH = DH.T
        # 矩阵做点乘,乘后调用.tocoo()方法保持稀疏矩阵
        DHBH_T = np.dot(DH,BH_T)

        return DHBH_T.tocoo(), mashup_data, g_m_d


    def get_mashup_adj(self, mashup_data):
        matrix = np.zeros((self.num_mashups, self.num_mashups))
        for i in range(self.num_mashups):
            mashup_a = set(mashup_data[i])
            for j in range(i + 1, self.num_mashups):
                mashup_b = set(mashup_data[j])
                overlap = mashup_a.intersection(mashup_b)
                ab_set = mashup_a | mashup_b
                matrix[i][j] = float(len(overlap) / len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0] * self.num_mashups)
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0 / degree)
        return matrix, degree

    def get_train_instances(self, train):
        user_input, pos_service_input, neg_service_input = [], [], []
        num_users = train.shape[0]
        num_services = train.shape[1]
        for (u, i) in train.keys():
            # positive instance
            for _ in range(self.num_negatives):
                pos_service_input.append(i)
            # negative instances
            for _ in range(self.num_negatives):
                j = np.random.randint(num_services)
                while (u, j) in train:
                    j = np.random.randint(num_services)
                user_input.append(u)
                neg_service_input.append(j)
        pi_ni = [[pi, ni] for pi, ni in zip(pos_service_input, neg_service_input)]
        return user_input, pi_ni

    def get_user_dataloader(self, batch_size):
        user, posservice_negservice_at_u = self.get_train_instances(self.user_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(user), torch.LongTensor(posservice_negservice_at_u))
        user_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return user_train_loader

    def get_mashup_dataloader(self, batch_size):
        mashup, posservice_negservice_at_g = self.get_train_instances(self.mashup_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(mashup), torch.LongTensor(posservice_negservice_at_g))
        mashup_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return mashup_train_loader






