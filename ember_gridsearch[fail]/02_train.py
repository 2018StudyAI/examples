import argparse
import os
import uak
import sys
import subprocess
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def clear(data_dir):
	path_X = os.path.join(data_dir, "X.dat")
	path_y = os.path.join(data_dir, "y.dat")

	if os.path.isfile(path_X):
		os.remove(path_X)
	if os.path.isfile(path_y):
		os.remove(path_y)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--datadir", help="Features Directory", type=str)
	args = parser.parse_args()

	if not os.path.exists(args.datadir) or not os.path.isdir(args.datadir):
		parser.error("{} is not a directory".format(args.parser))
	
	parameter_popen = ['wc', '-l', os.path.join(args.datadir, 'features.jsonl')]
	resut = subprocess.Popen(parameter_popen, stdout=subprocess.PIPE, universal_newlines=True).communicate()[0]
	rows = int(resut.split(' ')[0])

	clear(args.datadir)
	uak.create_vectorized_features(args.datadir, rows)

	X, y = uak.read_vectorized_features(args.datadir, rows)
	
	#cross validation
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	#train
	lgbm_dataset = lgb.Dataset(X_train, y_train)
	lgbm_model = lgb.train({"application": "binary"}, lgbm_dataset)

	#predict
	predictions_lgbm_prob = lgbm_model.predict(X_test)
	predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.75, 1, 0)

	#print accuracy
	acc_lgbm = accuracy_score(y_test, predictions_lgbm_01)
	print("accuaracy : ", acc_lgbm)
	print()
	print()


	# RandomForest
	randomforest = RandomForestClassifier()
	r_model = randomforest.fit(X_train, y_train)
	score = r_model.score(X_test, y_test)
	print(score)
	
	# GridSearch for lightgbm
	# params = {
	# 	#'verbosity': [0], 
	# 	'learning_rate': [0.15, 0.2, 0.25, 0.3],  #default 0.1
	# 	'num_leaves': [30, 35, 40, 45, 50], #defaut 31
	# 	'boosting_type': ['gbdt'],
	# 	'objective': ['binary'],
	# 	'max_bin': [255], #default 255
	# 	'random_state': [501], #update from seed
	# 	'n_estimators': [100, 200, 250, 300, 350, 400, 450, 500]
	# }

	params = {
		#'verbosity': [0], 
		'learning_rate': 0.25,  #default 0.1
		'num_leaves': 35, #defaut 31
		'boosting_type': 'gbdt',
		'objective': 'binary',
		'max_bin': 255, #default 255
		'random_state': 501, #update from seed
		'n_estimators': 400
	}
	
	#lgb.LGBMClassifier(params)
	lgbm_dataset = lgb.Dataset(X_train, y_train)
	lgbm_valid = lgb.Dataset(X_test, y_test, reference=lgbm_dataset)
	model = lgb.train(params, lgbm_dataset, valid_sets=[lgbm_dataset, lgbm_valid])
	model.save_model(os.path.join(args.datadir, "model.txt")) #save model
	
	#predict
	#predictions_lgbm_prob = gbm.predict(X_test)
	#predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.75, 1, 0)

	#print accuracy
	#acc_lgbm = accuracy_score(y_test, predictions_lgbm_01)
	#print("accuaracy : ", acc_lgbm)
	#print()
	#print()


	print("Features importance...")
	gain = model.feature_importance('gain')
	ft = pd.DataFrame({'feature':model.feature_name(), 'split':model.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
	print(ft.head(25))

	#grid = GridSearchCV(clf, params, verbose=2, cv=3, n_jobs=1)
	#grid_model = grid.fit(X, y)
	#model = grid_model.best_estimator_

	#print(grid.best_params_)
	#print(grid.best_score_)
	
	# Train and save model
	#model.save_model(os.path.join(args.datadir, "model.txt")) 

	# cross validation
	# print("Training LightGBM model with cross validation")
	# lgbm_model = ember.cross_validation(args.datadir, rows)
	# lgbm_model.save_model(os.path.join(args.datadir, "model.txt")) #save model
	
if __name__=='__main__':
	main()
	print("Done")