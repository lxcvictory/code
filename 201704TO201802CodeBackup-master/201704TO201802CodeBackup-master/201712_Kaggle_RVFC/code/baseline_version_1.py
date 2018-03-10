#coding:utf-8

from datetime import date, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
# train
df_train = pd.read_csv(
    '../data/air_visit_data.csv',
    converters={'visitors': lambda u: np.log1p(
        float(u))},
    parse_dates=['visit_date']
)
print 'train shape',df_train.shape

# test
df_test = pd.read_csv(
    '../data/sample_submission.csv'
)

df_test['visit_date'] = pd.to_datetime(df_test['id'].map(lambda x: str(x).split('_')[2]))
df_test['air_store_id'] = df_test['id'].map(lambda x: '_'.join(x.split('_')[:2]))

del df_test['id']
df_test = df_test[df_train.columns]
print 'test shape',df_test.shape

# air_store_info
df_info = pd.read_csv(
    '../data/air_store_info.csv'
)
# date_info
df_date_info = pd.read_csv(
    '../data/date_info.csv',
    parse_dates=['calendar_date']
).rename({'calendar_date':'visit_date'})

################feature engineers######################

df_train = df_train.set_index(
    ["air_store_id", "visit_date"])[["visitors"]].unstack(
        level=-1).fillna(False)

df_train.columns = df_train.columns.get_level_values(1)

df_test = df_test.set_index(["air_store_id", "visit_date"])[["visitors"]].unstack(level=-1).fillna(False)
df_test.columns = df_test.columns.get_level_values(1)
df_test = df_test.reindex(df_test.index).fillna(False)
data = pd.concat([df_train, df_test], axis=1)

print data













def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
        "day_1_2017": get_timespan(data, t2017, 1, 1).values.ravel(),
        "mean_3_2017": get_timespan(data, t2017, 3, 3).mean(axis=1).values,
        "mean_7_2017": get_timespan(data, t2017, 7, 7).mean(axis=1).values,
        "mean_14_2017": get_timespan(data, t2017, 14, 14).mean(axis=1).values,
        "mean_30_2017": get_timespan(data, t2017, 30, 30).mean(axis=1).values,
    })

    if is_train:
        y = data[
            pd.date_range(t2017, periods=39)
        ].values
        return X, y
    return X
#
print("Preparing dataset...")
t2017 = date(2017, 2, 1)
X_l, y_l = [], []
for i in range(6):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2017 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l

X_val, y_val = prepare_dataset(date(2017, 3, 1))
#
X_test = prepare_dataset(date(2017, 4, 23), is_train=False)

print X_test
print X_val
print y_val

print X_test

print("Training and predicting models...")
params = {
    'num_leaves': 31,
    'objective': 'regression',
    'min_data_in_leaf': 300,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'l2',
    'num_threads': 4
}

MAX_ROUNDS = 500
val_pred = []
test_pred = []
cate_vars = []

for i in range(39):
    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
    )
    dval = lgb.Dataset(
        X_val, label=y_val[:, i], reference=dtrain,
    )
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=100
    )
    print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True
    )))
    val_pred.append(bst.predict(
        X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    test_pred.append(bst.predict(
        X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))

# print("Validation mse:", mean_squared_error(
#     y_val, np.array(val_pred).transpose()))

print("Making submission...")
y_test = np.array(test_pred).transpose()

df_preds = pd.DataFrame(
    y_test,index=data.index,
columns=pd.date_range("2017-04-23", periods=39)
).stack().to_frame("visitors")
df_preds.index.set_names(["air_store_id", "visit_date"], inplace=True)



df_preds = df_preds.reset_index()


print df_preds

# submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
# submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
df_preds.to_csv('lgb.csv', float_format='%.4f', index=None)