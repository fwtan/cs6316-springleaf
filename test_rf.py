from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import time

def sk_rf(train_X, train_Y, test_X, test_Y, 
	my_nr_estimators = 10, 
	my_criterion='gini', 
	my_max_depth=None, 
	my_min_samples_split=2, 
	my_min_samples_leaf=1, 
	my_min_weight_fraction_leaf=0.0, 
	my_max_features='auto', 
	my_max_leaf_nodes=None, 
	my_bootstrap=True, 
	my_oob_score=False, 
	my_n_jobs=1, 
	my_random_state=None, 
	my_verbose=0, 
	my_warm_start=False, 
	my_class_weight=None):

	# Start time
	startTime = time.time();
	clf = RandomForestClassifier(
		n_estimators=my_nr_estimators, 
		criterion=my_criterion, 
		max_depth=my_max_depth, 
		min_samples_split=my_min_samples_split, 
		min_samples_leaf=my_min_samples_leaf, 
		min_weight_fraction_leaf=my_min_weight_fraction_leaf, 
		max_features=my_max_features, 
		max_leaf_nodes=my_max_leaf_nodes, 
		bootstrap=my_bootstrap, 
		oob_score=my_oob_score, 
		n_jobs=my_n_jobs, 
		random_state=my_random_state, 
		verbose=my_verbose, 
		warm_start=my_warm_start, 
		class_weight=my_class_weight);

	clf.fit(train_X, train_Y); 
	accTrain = clf.score(train_X, train_Y);
	print "Random Forest: train accuracy %f"%accTrain;
	accTest = clf.score(test_X, test_Y);
	print "Random Forest: test accuracy %f"%accTest;
	pred_Y = clf.predict(test_X);
	auc = metrics.roc_auc_score(test_Y, pred_Y, average='macro', sample_weight=None);
	print "Random Forest: area under the ROC curve %f"%auc;
	# Stop time
	stopTime = time.time();
	print "Elapsed time (random forest): %f"%(stopTime - startTime);