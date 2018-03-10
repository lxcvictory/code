# -*- coding: UTF-8 -*-
from gen_feat import make_train_set
from gen_feat import make_test_set
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc 
import matplotlib.pyplot as plt
import pandas as pd
def xgboost_make_submission():
	train_start_date = '2016-03-31'
	train_end_date = '2016-04-10'
	test_start_date = '2016-04-10'
	test_end_date = '2016-04-16'
	sub_start_date = '2016-04-06'
	sub_end_date = '2016-04-16'

	user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
	list_of_train = list(training_data.columns)
	print len(list_of_train)

	X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2, random_state=0)
	dtrain=xgb.DMatrix(X_train, label=y_train)
	dtest=xgb.DMatrix(X_test, label=y_test)
	param = {'learning_rate' : 0.1, 'n_estimators': 1000, 'max_depth': 3, 
	'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
	'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
	# num_round = 345
	num_round = 511
	# param['nthread'] = 8
	param['eval_metric'] = "auc"
	plst = param.items()
	plst += [('eval_metric', 'logloss')]
	evallist = [(dtest, 'eval'), (dtrain, 'train')]
	bst=xgb.train(plst, dtrain, num_round, evallist,early_stopping_rounds=10)
	importance = bst.get_fscore()
	print importance
	feat_importances = []
	for ft,score in importance.iteritems():
		ft = ft.split('f')[1]
		feat_importances.append({'Feature': ft, 'Importance': score})
	feat_importances = pd.DataFrame(feat_importances)
	feat_importances = feat_importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)
	new_columns = []
	for index in list(feat_importances['Feature']):
		index = int(index)
		# feat_importances[index]['Feature'] = 
		new_columns.append(list_of_train[index])
	name_of = pd.DataFrame({'new':new_columns})
	feat_importances = pd.concat([feat_importances,name_of],axis=1)
	feat_importances.to_csv('./sub/fecure.csv')
	sub_user_index, sub_trainning_data = make_test_set(sub_start_date, sub_end_date)
	sub_trainning_data = xgb.DMatrix(sub_trainning_data.values)

	y_label = bst.predict(xgb.DMatrix(X_test))
	fpr,tpr,threasholds = roc_curve(y_test,y_label,pos_label=2)
	print fpr,tpr,threasholds
	# print auc(fpr,tpr)
	# plt.plot(threasholds,fpr)
	# plt.show()
	y = bst.predict(sub_trainning_data)
	sub_user_index['label'] = y
	# print np.median(y)
	# print sub_user_index
	# pred = sub_user_index.groupby('user_id').max().reset_index()
	# print pred
	pred = sub_user_index[sub_user_index['label'] >= 0.04]

	pred = pred[['user_id', 'sku_id']]
	pred = pred.groupby('user_id').max().reset_index()
	
	pred['user_id'] = pred['user_id'].astype(int)
	pred.to_csv('./sub/submission.csv', index=False, index_label=False)
	buy_cate_8 = np.load('./unique_8/user_id_unique.npy')
	pred = pred[~pred['user_id'].isin(buy_cate_8)]
	pred.to_csv('./sub/submission_unique.csv', index=False, index_label=False)

if __name__ == '__main__':

    xgboost_make_submission()
