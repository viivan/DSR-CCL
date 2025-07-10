class Config(object):
    def __init__(self):
        # 使用从programmableWeb网站爬取的Web数据
        self.path = './datasets/serviceData/'
        self.user_service = self.path + 'userService'
        self.user_service_compose = self.path + 'userCompose'
        self.service_mashup_path = self.path + 'serviceCompose.txt'
        # d 代表embedding 的特征维度dimension
        self.d = 32
        self.epoch = 201
        self.user_epoch = 10
        self.num_negatives = 10
        self.layers = 2
        self.batch_size = 512
        # self.lr = [0.000005, 0.000001, 0.0000005]
        self.lr = [0.0001, 0.00005, 0.00002]
        self.drop_ratio = 0.5
        self.topK = [10, 20]
        self.balance = 6
        self.gpu_id = 0
        self.behavior = 3

