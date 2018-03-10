import pandas as pd
from sklearn.metrics import mean_squared_error
x = pd.read_csv('../result/d_answer_b_20180130.csv',header=None)
y = pd.read_csv('../code/sub20180130.csv',header=None)

print(mean_squared_error(x[0],y[0])/2)
# print(
#     x[0] - y[0]
# )
#
#
# merge_x_y = (x[0] + y[0]) / 2.0
# merge_x_y.to_csv('../result/10点后提交.csv', float_format='%.3f', index=None,header=None)
# print(merge_x_y.describe())
