import load_springleaf
import numpy as np
from test_svm import test_svm 
from sklearn import preprocessing
from test_logistic_reg import tf_logistic_reg, sk_logistic_reg
from test_nn import tf_nn
from test_rf import sk_rf
import time
import pandas as pd
from dateutil.tz import tzutc
from dateutil.parser import parse, parserinfo

if __name__ == '__main__':

     # Start time
     startTime = time.time();

     # load dataframe
     dat_frame = load_springleaf.load("../data/train.csv");

     # # consolidation
     feat_mask, feat_dict, con_X, con_Y  = load_springleaf.consolidation(dat_frame);
     con_Y = np.squeeze(con_Y).astype(np.float64);

     # imputation
     imputed_X = load_springleaf.impute(con_X);
     
     # one hot encode
     ext_X = load_springleaf.encode_cate_feats(imputed_X, feat_mask);
     normalized_X = load_springleaf.normalize(ext_X);


     # # # total_XY = np.concatenate((ext_X, raw_Y), axis = 1);
     # # # total_XY_normalized = np.concatenate((normalized_X, raw_Y), axis = 1);

     # # # dat_XY = pd.DataFrame(total_XY);
     # # # dat_XY_normalized = pd.DataFrame(total_XY_normalized);

     # # # dat_XY.to_csv(path_or_buf="train_numeric.csv");
     # # # dat_XY_normalized.to_csv(path_or_buf="train_numeric_normalized.csv");

     # # raw_Y = np.squeeze(raw_Y).astype(np.float64);

     total_X, total_Y = load_springleaf.shuffle_XY(normalized_X, con_Y);
     train_X, train_Y, test_X, test_Y = load_springleaf.split(total_X, total_Y, 0.7);

     # Stop time
     stopTime = time.time();
     print "Elapsed time (load data): %f"%(stopTime - startTime);

     # # svm
     # sk_rf(train_X, train_Y, test_X, test_Y);
     # sk_logistic_reg(train_X, train_Y, test_X, test_Y);
     # test_svm(1.0, 'linear', 1.0, 0.0, 3, train_X, train_Y, test_X, test_Y);
     # logistic regression
     # tf_logistic_reg(train_X, train_Y, test_X, test_Y);
     
     # neural network
     tf_nn(train_X, train_Y, test_X, test_Y);


     


     
     

    

