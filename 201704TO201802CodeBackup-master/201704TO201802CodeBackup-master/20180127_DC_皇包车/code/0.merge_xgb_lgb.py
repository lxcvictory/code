import pandas as pd

xgb = pd.read_csv('../result/result_20180208_1.csv')
lgb = pd.read_csv('../result/9667.csv')


xgb_lgb = pd.merge(xgb,lgb,on=['userid'])
xgb_lgb['orderType'] = (xgb_lgb['orderType_x'] * 0.5 + xgb_lgb['orderType_y'] * 0.5)
print(xgb_lgb)

xgb_lgb[['userid','orderType']].to_csv('../result/result_finish_this_company_thank_my_parten.csv' ,index=False)