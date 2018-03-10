import pandas as pd
from sklearn.metrics import mean_squared_error as mse

wsp = pd.read_csv('../tmp/wqs_2017-12-21-11-02-16-0.0432.csv',header=None)
lgb = pd.read_csv('../result/result_20171225.csv',header=None)
result = pd.DataFrame()

result['ID'] = wsp[0]
result['Y'] = 2.0 / (1.0 / wsp[1] + 1.0/lgb[1])

result[['ID','Y']].to_csv('../result/result_20171227_h_lgb_xgb.csv',index=False,header=False)
