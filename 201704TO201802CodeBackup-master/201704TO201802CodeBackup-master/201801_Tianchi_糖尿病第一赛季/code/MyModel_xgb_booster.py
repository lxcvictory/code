import xgboost as xgb
import numpy as np
import pandas as pd

class my_model:
    def __init__(self):
        # 读取数据
        self.train =  pd.read_csv('../data/d_train_20180102.csv',encoding='gbk')
        self.test =  pd.read_csv('../data/d_test_A_20180102.csv',encoding='gbk')
        # 模型初始化
        self.model= xgb.XGBRegressor(
                             gamma=0.0468,
                             learning_rate=0.05,
                             max_depth=5,
                             n_estimators=2200,
                             reg_alpha=0.4640,
                             reg_lambda=0.8571,
                             subsample=0.5213,
                             nthread = -1,
        )
        # l等是模型的参数，需要在get_params返回，类似于树的深度，模型的随机种子等等
        self.l = self.model.get_params()

    def fit(self, X, y):

        self.model.fit(X,y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=False):
        # 返回模型的参数
        return {'l': self.l}



if __name__ == '__main__':
    m = my_model()
    print(m.get_params())