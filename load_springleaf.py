import numpy as np
import pandas as pd
from dateutil.tz import tzutc
from dateutil.parser import parse
from sklearn import preprocessing
import random as randomGen

def diff_days(s):
    cur_time = parse(s,            yearfirst=True, tzinfos=tzutc);
    min_time = parse('1970-01-01', yearfirst=True, tzinfos=tzutc);
    diff = cur_time - min_time;
    return diff.days;

def load(path):
    
    datFrame = pd.read_csv(path,  
                     header=0, 
                     index_col=0, # can be modified
                     names=None, 
                     prefix=None, 
                     skiprows=None, 
                     skipfooter=None, 
                     skip_footer=0, 
                     na_values=None, 
                     true_values=None, 
                     false_values=None, 
                     delimiter=None, 
                     converters=None, 
                     dtype=None, 
                     usecols=None, # can be modified
                     engine='c', 
                     na_filter=True,  
                     warn_bad_lines=True, 
                     error_bad_lines=False, 
                     keep_default_na=True, 
                     thousands=None, 
                     comment='#', 
                     decimal='.', 
                     nrows=None, # can be modified
                     iterator=False, # can be modified
                     chunksize=None, # can be modified
                     verbose=False, 
                     encoding=None, 
                     squeeze=False, 
                     mangle_dupe_cols=True, 
                     tupleize_cols=False, 
#                     parse_dates=[0], 
                     parse_dates=False,
                     keep_date_col=False, 
                     dayfirst=False, 
 #                    date_parser=date_utc, 
                     date_parser=None,
                     infer_datetime_format=False);

    # Write summaries of the train and test sets to the log
    print('\nSummary and memory usage of train dataset:\n')
    print(datFrame.info(memory_usage = True))
    return datFrame;
    
# initialize the dictionary and mask for categorical features
def consolidation(dat_frame):
    raw_data  = dat_frame.values;
    raw_types = dat_frame.dtypes;

    nr_cols = raw_data.shape[1];  
    
    raw_X = raw_data[:, 0:nr_cols-1];
    raw_Y = raw_data[:, nr_cols-1:nr_cols];
    
    raw_Y = 2 * raw_Y - 1;
    
    # True: categorical features, False: continuous features
    feat_mask = map(lambda x: not (x == 'float64' or x == 'int64'), raw_types[0:nr_cols-1]);
    # Collect all possible values for each categorical features
    feat_dict = map(lambda x: dict.fromkeys(x), np.transpose(raw_X).tolist());
    
    # datetime data
#    for i in xrange(15, 18):
#        feat_mask[i] = False;
        
    # Assign a unique positive integer to a categorical value and convert missing data to 0
    for i in xrange(0, len(feat_dict)):
        if feat_mask[i]:
            j = 0;
            for k in feat_dict[i].keys():
                if k != "-1":
                    j += 1;
                    feat_dict[i][k] = j;  
                else:
                    feat_dict[i]["-1"] = 0; 
    # from categorical to numerical
    raw_vals = np.zeros(raw_X.shape);    
    for x in xrange(0, raw_vals.shape[0]):
        for y in xrange(0, raw_vals.shape[1]):
#            if y < 18 and y > 14:
#                raw_vals[x][y] = diff_days(raw_X[x][y]);               
#            elif feat_mask[y]:
            if feat_mask[y]:
                raw_vals[x][y] = feat_dict[y][raw_X[x][y]];
#            elif raw_data[x][y] == "-1":
#                raw_vals[x][y] = np.nan;
            else:
                raw_vals[x][y] = np.float32(raw_X[x][y]);

    return feat_mask, feat_dict, raw_vals, raw_Y;
    
def impute(raw_vals):
    imp = preprocessing.Imputer(missing_values='NaN', strategy='median', axis=0, verbose=0, copy=True);
    imputed_vals = imp.fit_transform(raw_vals);
    return imputed_vals; 
 
def encode_cate_feats(raw_vals, feat_mask):
    enc = preprocessing.OneHotEncoder(n_values='auto', categorical_features=feat_mask);
    ext_vals = enc.fit_transform(raw_vals).toarray();
    return ext_vals;
    
def normalize(ext_vals):  
    normalized_vals = (ext_vals - np.mean(ext_vals, axis=0))/np.std(ext_vals, axis=0);
    return normalized_vals;

def shuffle_XY(X, Y, randSeed = 37):
    nrSamples  = X.shape[0];
    randomInds = range(0, nrSamples);
    randomGen.seed(randSeed);
    randomGen.shuffle(randomInds);
    
    sX = X[randomInds];
    sY = Y[randomInds];
    
    return sX, sY;

def split(X, Y, ratio = 0.7):
    X, Y = shuffle_XY(X, Y, randSeed = 53);
    nrSamples  = X.shape[0];
    nrTrain = np.int32(np.floor(nrSamples * ratio));

    train_X = X[0:nrTrain-1];
    train_Y = Y[0:nrTrain-1];

    test_X = X[nrTrain:nrSamples-1];
    test_Y = Y[nrTrain:nrSamples-1];

    return train_X, train_Y, test_X, test_Y;

def sample(X, Y, offset, batch_size=256):
    nr_samples = X.shape[0];
    first = offset;
    last  = offset + batch_size;

    if last > nr_samples:
        new_X = np.concatenate((X[first:last], X[0:(last%nr_samples)]), axis = 0);
        new_Y = np.concatenate((Y[first:last], Y[0:(last%nr_samples)]), axis = 0);
    else:
        new_X = X[first:last];
        new_Y = Y[first:last];

    return new_X, new_Y, last%nr_samples;



    