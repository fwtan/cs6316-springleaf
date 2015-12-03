import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from load_springleaf import sample
import time
from sklearn import metrics

# declaration for W
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1);
    return tf.Variable(initial);

# declaration for b
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape);
    return tf.Variable(initial);

def tf_nn(train_X, train_Y, test_X, test_Y):
	# Start time
	startTime = time.time();
	enc = preprocessing.OneHotEncoder(n_values='auto');
	train_Y_onehot = enc.fit_transform(np.int32((np.transpose(np.matrix(train_Y)) + 1)/2)).toarray();
	test_Y_onehot = enc.fit_transform(np.int32((np.transpose(np.matrix(test_Y)) + 1)/2)).toarray();

	nr_vars = train_X.shape[1];
	x  = tf.placeholder("float", [None, nr_vars]);
	y_ = tf.placeholder("float", [None, 2]);
	
	# instance of Ws and bs

	W_fc1 = weight_variable([nr_vars, 1024]);
	b_fc1 = bias_variable([1024]);

	W_fc3 = weight_variable([1024, 257]);
	b_fc3 = bias_variable([257]);

	W_fc2 = weight_variable([257, 2]);
	b_fc2 = bias_variable([2]);

	# operations
	h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1);
#	keep_prob = tf.placeholder("float");
#	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob);
#	y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2);
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc3) + b_fc3);
	y = tf.nn.softmax(tf.matmul(h_fc2, W_fc2) + b_fc2);

	cross_entropy = -tf.reduce_sum(y_ * tf.log(y));
	train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy);
#	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	init = tf.initialize_all_variables();
	sess = tf.Session();
	sess.run(init);

	cur_id = 0
	for i in range(20000):
		print i;
		batch_xs, batch_ys, cur_id = sample(train_X, train_Y_onehot, cur_id, 256);
    	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys});

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1));
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"));
	# sess.run(tf.initialize_all_variables());

	# cur_id = 0;
	# for i in range(20000):
	# 	batch_xs, batch_ys, cur_id = sample(train_X, train_Y_onehot, cur_id, 256);
	# 	if i%100 == 0:
	# 		train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
	# 		print "step %d, training accuracy %f"%(i, train_accuracy)
	# 	train_step.run(feed_dict={x: batch_xs, y_: batch_ys})

	print "NN: train accuracy %f"%sess.run(accuracy, feed_dict={x: train_X, y_: train_Y_onehot});
	print "NN: test accuracy %f"%sess.run(accuracy, feed_dict={x: test_X, y_: test_Y_onehot});
#	auc = metrics.roc_auc_score(test_Y, pred_Y, average='macro', sample_weight=None);
#	print "NN: area under the ROC curve %f"%auc;

	# Stop time
	stopTime = time.time();
	print "Elapsed time (neural network): %f"%(stopTime - startTime);