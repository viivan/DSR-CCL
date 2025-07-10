import random
import numpy as np
import  math

import torch

print("初始化训练数据")
# 获取训练样本
#前10项API服务
k = 100
file = open("serviceTest.txt")
apis = []
for line in file:
    api = list(map(int,line.split()))[1:]
    apis.append(api)
# 转化为二维数组
# 创建示例的二维数组

# 获取所有不重复的值
def getPopAPI(apiInput):
    uniqueAPI, count = np.unique(apiInput, return_counts=True)
    # 计算每个值在原始数组中出现的次数
    apiDict = {}
    for i in range(len(uniqueAPI)):
        apiDict.setdefault(uniqueAPI[i], count[i])
    sortedAPI = sorted(apiDict.services(), key=lambda x: x[1], reverse=True)
    result = []
    for i in range(k):
        result.append(sortedAPI[i][0])
    return result
#训练样本


def hit(apiList1, apiList2,m):
    apiList1 = apiList1[0:m]
    apiList2 = apiList2[0:m]
    hit = 0
    for a1 in apiList1:
        for a2 in apiList2:
            if(a1 == a2):
                hit =hit+1
    return hit/m
def getNDCG(apiList1, apiList2,m):
    apiList1 = apiList1[0:m]
    apiList2 = apiList2[0:m]
    ndcgs = []
    for a1 in apiList1:
        j = 0
        for a2 in apiList2:
            j+=1
            if (a1 == a2):
                ndcgs.append(math.log(2) / math.log(j + 2))
    if ndcgs!=[]:
        return np.mean(ndcgs)
    else:
        return 0

def mrr(apiList1,apiList2):
    mrr = 0
    # 遍历所有推荐结果列表
    for a2 in apiList2:
        i = 0
        #寻找在groundTruth中的排名
        for a1 in apiList1:
            i += 1
            if (a1 == a2):
                mrr+=1/i
    return mrr/20
#重复10次结果，取平均值：

def getILS(apiList1,apiList2):
    for api in apiList1:
        for api2 in apiList1:
            a = torch.randint(api,[1,32])
            b = torch.randint(api,[1,32])
            c = torch.cosine_similarity(a,b).float()
            print(c)

if __name__ == "__main__":
    HR10 = []
    HR20 = []
    MRR = []
    ILS =[]
    NDCG10 = []
    NDCG20 =[]
    for j in range(0,5):
        for i in range(0, 100):
            trainData = random.sample(apis, int(len(apis) * 0.8))
            # 获取测试样本
            uniqueAPI = np.unique(apis)
            test = random.sample(apis, int(len(apis) * 0.05))
            # print(testData)
            # 找出评率最高的API服务
            real = getPopAPI(trainData)

            test = getPopAPI(test)
            # 打印结果
            HR10.append(hit(real, test, 10))
            HR20.append(hit(real, test, 20))
            NDCG10.append(getNDCG(real, test, 10))
            NDCG20.append(getNDCG(real, test, 20))
            ILS.append(getILS(real,test))
            MRR.append(mrr(real, test))
        print(f"HR@10 : {np.mean(HR10)} HR@20 : {np.mean(HR20)} NDCG@10 : {np.mean(NDCG10)},NDCG@20 : {np.mean(NDCG20)} MRR : {np.mean(MRR)}  ILS : {np.mean(ILS)}")

#进行指标计算


