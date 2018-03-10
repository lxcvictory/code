import pandas as pd

x = pd.read_csv('../result/result_0111_3.csv',header=None)
y = pd.read_csv('../code/sub20180111.csv',header=None)

x['res'] = (x[0] + y[0])/2
x['res'].to_csv('./20180111_1.csv',float_format='%.3f',header=False,index=False)
print(x.describe())