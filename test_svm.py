
from sklearn import svm, metrics
import time

def test_svm(myC, myKernel, myGamma, myCoef0, myDegree, train_X, train_Y, test_X, test_Y):
	# Start time
	startTime = time.time();
	clf = svm.SVC(
		C=myC, 
		kernel=myKernel, 
		gamma=myGamma, 
		coef0=myCoef0, 
		degree=myDegree, 
		probability=False,
		shrinking=True, 
		tol=0.001, 
		cache_size=2000, 
		class_weight=None,
		verbose=False,   
		max_iter=-1,  
		random_state=None);
	clf.fit(train_X, train_Y); 
	accTrain = clf.score(train_X, train_Y);
	print "SVM: train accuracy %f"%accTrain;
	accTest = clf.score(test_X, test_Y);
	print "SVM: test accuracy %f"%accTest;
	pred_Y = clf.predict(test_X);
	auc = metrics.roc_auc_score(test_Y, pred_Y, average='macro', sample_weight=None);
	print "SVM: area under the ROC curve %f"%auc;
	# Stop time
	stopTime = time.time();
	print "Elapsed time (svm): %f"%(stopTime - startTime);