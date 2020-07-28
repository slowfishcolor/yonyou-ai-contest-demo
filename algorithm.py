import lightgbm as lgb
import numpy as np
class Algorithm:
    """
固定，名称不能改动
    算法类， 导入的算法必须定义class Algorithm， 并放入 algorithm.py 文件中
    """

    def __init__(self, parameters):
        """
        初始化， parameters 为 dict格式的参数信息， 比如
        {
            "params1": "value1",
            "params1", 2
        }
        """
        self.params = {
            'num_leaves': 80,
            'objective': 'regression',
            'min_data_in_leaf': 200,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 1,
            'metric': 'l2',
            'num_threads': 16
        }

    def load(self, path):
        """
        加载模型
            Args:
                path : 模型文件路径，根据 model.yaml 文件中的 location 字段指定
            Returns:
                无
        """
        self.gbm = lgb.Booster(model_file=path)

    def save(self, path):
        """
        保存模型， 需要使用此方法来保存模型
            Args:
                path : 模型文件路径， 文件名或文件夹名，根据算法自定义
            Returns:
                无
        """
        self.gbm.save_model(path)

    def train(self, train_reader, val_reader):
        """
        模型训练
            Args:
                train_reader : 训练数据读取接口
                val_reader : 验证数据读取接口
            Returns:
                无
        """
        x_train = []
        y_train = []
        x_val = []
        y_val = []
        for train_data in train_reader:
            if train_data is not None:
                x_train.append(train_data[0:2]+train_data[3:6])
                y_train.append(float(train_data[6]))
        for val_data in val_reader:
            if val_data is not None:
                x_val.append(train_data[0:2]+train_data[3:6])
                y_val.append(float(val_data[6]))
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)

        self.gbm = lgb.train(self.params,
                             lgb_train,
                             num_boost_round=200,
                             valid_sets=lgb_eval,
                             early_stopping_rounds=125, verbose_eval=50)
        print('训练完成')
        self.save('./gbm.model')
        print('模型保存')


    def infer(self, test_data):
        """
        推理
            Args:
                test_data : 推理请求数据, json格式， 定义为
                {
                    "data": # 推理请求数据
                }
            Returns:
                推理结果, json
                {
                    "data"： 结果数据，任意格式# 推理结果
                }
        """
        assert ('data' in test_data)
        return self.gbm.predict(test_data['data'], num_iteration=self.gbm.best_iteration)
