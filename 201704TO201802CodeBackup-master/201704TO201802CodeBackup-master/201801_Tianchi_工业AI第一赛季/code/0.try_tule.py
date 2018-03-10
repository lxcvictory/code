#coding:utf-8

import pandas as pd
import numpy as np
import warnings
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error as mse

fileDir = '../result/'
train = pd.read_csv(fileDir + u"zs_2018-01-14-22-18-10.csv",header=None)
print train.describe()

