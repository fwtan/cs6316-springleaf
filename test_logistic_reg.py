import tensorflow as tf
import numpy as np
from sklearn import preprocessing, metrics
from load_springleaf import sample
import time
from sklearn.linear_model import LogisticRegression

def sk_logistic_reg(train_X, train_Y, test_X, test_Y,
	my_penalty='l2', 
	my_dual=False, 
	my_tol=0.0001, 
	my_C=1.0, 
	my_fit_intercept=True, 
	my_intercept_scaling=1, 
	my_class_weight=None, 
	my_random_state=None, 
	my_solver='liblinear', 
	my_max_iter=100, 
	my_multi_class='ovr', 
	my_verbose=0, 
	my_warm_start=False, 
	my_n_jobs=1):

	# Start time
	startTime = time.time();
	clf = LogisticRegression(
		penalty=my_penalty, 
		dual=my_dual, 
		tol=my_tol, 
		C=my_C, 
		fit_intercept=my_fit_intercept, 
		intercept_scaling=my_intercept_scaling, 
		class_weight=my_class_weight, 
		random_state=my_random_state, 
		solver=my_solver, 
		max_iter=my_max_iter, 
		multi_class=my_multi_class, 
		verbose=my_verbose, 
		warm_start=my_warm_start, 
		n_jobs=my_n_jobs);

	clf.fit(train_X, train_Y); 
	accTrain = clf.score(train_X, train_Y);
	print "Logistic Regression: train accuracy %f"%accTrain;
	accTest = clf.score(test_X, test_Y);
	print "Logistic Regression: test accuracy %f"%accTest;
	pred_Y = clf.predict(test_X);
	auc = metrics.roc_auc_score(test_Y, pred_Y, average='macro', sample_weight=None);
	print "Random Forest: area under the ROC curve %f"%auc;
	# Stop time
	stopTime = time.time();
	print "Elapsed time (logistic regression): %f"%(stopTime - startTime);


def tf_logistic_reg(train_X, train_Y, test_X, test_Y):
	# Start time
	startTime = time.time();
	enc = preprocessing.OneHotEncoder(n_values='auto');
	train_Y_onehot = enc.fit_transform(np.int32((np.transpose(np.matrix(train_Y)) + 1)/2)).toarray();
	test_Y_onehot = enc.fit_transform(np.int32((np.transpose(np.matrix(test_Y)) + 1)/2)).toarray();

	x  = tf.placeholder("float", [None, train_X.shape[1]]);
	y_ = tf.placeholder("float", [None, 2]);

	W = tf.Variable(tf.zeros([train_X.shape[1], 2]));
	b = tf.Variable(tf.zeros([2]));
	y = tf.nn.softmax(tf.matmul(x,W) + b);
	cross_entropy = -tf.reduce_sum(y_*tf.log(y));
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy);

	init = tf.initialize_all_variables();
	sess = tf.Session();
	sess.run(init);

	cur_id = 0;
	for i in range(1000):
		batch_xs, batch_ys, cur_id = sample(train_X, train_Y_onehot, cur_id, 500);
    	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys});
    
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1));
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"));
	accu_train = sess.run(accuracy, feed_dict={x: train_X, y_: train_Y_onehot});
	accu_test = sess.run(accuracy, feed_dict={x: test_X, y_: test_Y_onehot});
	print "Logistic Regression: train accuracy %f"%accu_train;
	print "Logistic Regression: test accuracy %f"%accu_test;
	# Stop time
	stopTime = time.time();
	print "Elapsed time (logistic regression): %f"%(stopTime - startTime);
