import numpy as np
import pandas as pd
from dateutil.tz import tzutc
from dateutil.parser import parse
from sklearn import preprocessing
import random as randomGen
from collections import Counter

default_min_time = "01-01-1970";
valid_idx = [1, 2, 5, 51, 65, 75, 85, 93, 96, 104, 105, 113, 117, 120, 143, 145, 170, 201, 204, 217, 224, 227, 228, 231, 232, 233, 234, 235, 237, 238, 241, 251, 255, 256, 258, 263, 272, 274, 279, 280, 281, 282, 283, 285, 286, 287, 290, 291, 292, 293, 294, 296, 297, 298, 302, 304, 305, 306, 308, 310, 311, 313, 314, 315, 316, 317, 318, 323, 325, 328, 330, 331, 332, 336, 337, 339, 340, 342, 344, 345, 346, 347, 348, 349, 351, 352, 353, 354, 358, 361, 362, 363, 364, 375, 376, 382, 417, 418, 424, 426, 434, 454, 464, 466, 467, 473, 482, 490, 513, 515, 520, 537, 538, 539, 541, 547, 548, 560, 590, 594, 597, 605, 606, 608, 609, 616, 621, 624, 637, 640, 646, 659, 674, 685, 691, 692, 694, 711, 712, 714, 715, 719, 723, 731, 734, 735, 736, 737, 743, 760, 773, 776, 785, 787, 797, 799, 800, 801, 802, 804, 805, 806, 807, 808, 809, 812, 814, 815, 817, 826, 829, 830, 831, 832, 833, 834, 835, 842, 846, 848, 854, 857, 858, 859, 868, 871, 881, 882, 884, 888, 889, 899, 906, 909, 917, 922, 927, 928, 933, 934, 940, 941, 945, 946, 948, 952, 953, 961, 970, 988, 999, 1000, 1001, 1058, 1063, 1071, 1075, 1077, 1080, 1081, 1088, 1089, 1096, 1100, 1104, 1113, 1116, 1118, 1119, 1123, 1134, 1137, 1143, 1146, 1149, 1150, 1187, 1189, 1190, 1212, 1240, 1242, 1244, 1245, 1246, 1259, 1321, 1371, 1372, 1379, 1382, 1397, 1409, 1411, 1412, 1415, 1432, 1482, 1483, 1485, 1487, 1506, 1509, 1510, 1511, 1515, 1518, 1520, 1523, 1525, 1526, 1527, 1532, 1535, 1537, 1541, 1542, 1549, 1552, 1554, 1555, 1615, 1616, 1629, 1632, 1633, 1634, 1683, 1684, 1685, 1686, 1687, 1691, 1692, 1694, 1699, 1706, 1709, 1715, 1723, 1725, 1727, 1740, 1743, 1798, 1828, 1830, 1836, 1839, 1886, 1894, 1896, 1903, 1904, 1906, 1907, 1908, 1909, 1910, 1911, 1916, 1917, 1918, 1920, 1921, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1934];

def idx2name(id):
    x = "VAR_%04d"%id;
    return x;

def str2date(in_ds):
    if in_ds == "nan":
        return default_min_time;
    dd = in_ds[0:2];
    mm = in_ds[2:5];
    yy = in_ds[5:7];
    return mm+"-"+dd+"-"+"20"+yy;

def diff_days(in_ds):
    cur_time = parse(in_ds);
    min_time = parse(default_min_time);
    diff = cur_time - min_time;
    return diff.days;

def date_utc(s):
    cur_time = parse(s, tzinfos=tzutc);
    return cur_time;

def load(path):
    # offset = 0;
    # dt_idx = [73+offset, 75+offset, 156+offset, 157+offset, 158+offset, 159+offset, 166+offset, 167+offset, 168+offset, 169+offset, 176+offset, 177+offset, 178+offset, 179+offset, 204+offset, 217+offset];

    valid_names = map(idx2name, valid_idx);
    valid_names.append("target");
    datFrame = pd.read_csv(path,  
#       parse_dates=xrange(0,16), 
        parse_dates=False,
        keep_date_col=False, 
        dayfirst=None, 
#       date_parser=diff_days, 
        date_parser=None,
        infer_datetime_format=False,
        header=0, 
        index_col=None,   # can be modified
        nrows=None,    # can be modified
        skiprows=None, 
        usecols=valid_names,  # can be modified
        names=None, 
        prefix=None,   # column prefix 
        true_values=None, 
        false_values=None, 
        na_filter=True, 
        na_values=['[]'], 
        delimiter=None, 
        converters=None, 
        dtype=None, 
        engine='c',  
        skipfooter=None, 
        warn_bad_lines=True, 
        error_bad_lines=False, 
        keep_default_na=True, 
        thousands=None, 
        comment='#', 
        decimal='.', 
        iterator=False, # can be modified
        chunksize=None, # can be modified
        verbose=False, 
        encoding=None, 
        squeeze=False, 
        mangle_dupe_cols=True, 
        tupleize_cols=False);

    # del datFrame['VAR_0008'];
    # del datFrame['VAR_0009'];
    # del datFrame['VAR_0010'];
    # del datFrame['VAR_0011'];
    # del datFrame['VAR_0012'];
    # del datFrame['VAR_0043'];
    # del datFrame['VAR_0073'];
    # del datFrame['VAR_0156'];
    # del datFrame['VAR_0157'];
    # del datFrame['VAR_0158'];
    # del datFrame['VAR_0159'];
    # del datFrame['VAR_0166'];
    # del datFrame['VAR_0167'];
    # del datFrame['VAR_0168'];
    # del datFrame['VAR_0169'];
    # del datFrame['VAR_0176'];
    # del datFrame['VAR_0177'];
    # del datFrame['VAR_0178'];
    # del datFrame['VAR_0179'];
    # del datFrame['VAR_0196'];
    # del datFrame['VAR_0202'];
    # del datFrame['VAR_0214'];
    # del datFrame['VAR_0216'];
    # del datFrame['VAR_0222'];
    # del datFrame['VAR_0226'];
    # del datFrame['VAR_0229'];
    # del datFrame['VAR_0230'];
    # del datFrame['VAR_0236'];
    # del datFrame['VAR_0239'];
    # del datFrame['VAR_0404'];
    # del datFrame['VAR_0493'];


    # del datFrame['VAR_0200'];
    # del datFrame['VAR_0207'];
    # del datFrame['VAR_0213'];
    # del datFrame['VAR_0237'];
    # del datFrame['VAR_0840'];
    # del datFrame['VAR_0847'];
    # del datFrame['VAR_1428'];

  #  print valid_names;
    # Write summaries of the train and test sets to the log
    print('\nSummary and memory usage of train dataset:\n')
    print(datFrame.info())
    return datFrame;

def clean_data_frame(dat_frame):
    raw_data  = dat_frame.values;
    raw_types = dat_frame.dtypes;
    nr_rows   = raw_data.shape[0]; 
    nr_cols   = raw_data.shape[1]; 

    # True: categorical features, False: continuous features
    feat_mask = map(lambda x: not (x == 'float64' or x == 'int64'), raw_types[0:nr_cols-1]);
    print "freq";
    # Collect all possible values for each categorical features
    feat_freq = map(lambda x: Counter(x), np.transpose(raw_data[:,0:nr_cols-1]).tolist());
    print "dict";
    feat_dict = map(dict, feat_freq);

    feat_stat = [sorted(dict(feat_freq[y].most_common(2)).values(), reverse=True) for y in xrange(0, len(feat_dict))];
    feat_outl1 = [str(dat_frame.columns[y]) for y in xrange(0, len(feat_dict)) if feat_mask[y] and (feat_stat[y][0] > 19 * feat_stat[y][1])]

    for i in xrange(0, len(feat_outl1)):
          del dat_frame[feat_outl1[i]];

    return feat_outl1, dat_frame;

    
# initialize the dictionary and mask for categorical features
def consolidation(dat_frame):
    dt_idx = [5, 18, 19];

    raw_data  = dat_frame.values;
    raw_types = dat_frame.dtypes;
    nr_rows   = raw_data.shape[0]; 
    nr_cols   = raw_data.shape[1];  
    con_Y = raw_data[:, nr_cols-1:nr_cols];
    con_Y = 2 * con_Y - 1;
    
    # True: categorical features, False: continuous features
    feat_mask = map(lambda x: not (x == 'float64' or x == 'int64'), raw_types[0:nr_cols-1]);
    # Collect all possible values for each categorical features
    feat_dict = map(lambda x: dict.fromkeys(x), np.transpose(raw_data[:,0:nr_cols-1]).tolist());

    # Assign a unique positive integer to a categorical value and convert missing data to 0
    for i in xrange(0, len(feat_dict)):
        if feat_mask[i]:
            j = 0;
            for k in feat_dict[i].keys():
                if k != "nan":
                    j += 1;
                    feat_dict[i][k] = j;  
                else:
                    feat_dict[i]["nan"] = 0; 

    # from categorical to numerical
    con_X = np.zeros([nr_rows, nr_cols-1]);    
    for x in xrange(0, nr_rows):
        for y in xrange(0, nr_cols-1):
            if y in dt_idx:
                dt_str = str2date(str(raw_data[x][y]));
                con_X[x][y] = diff_days(dt_str);             
            elif feat_mask[y]:
                con_X[x][y] = feat_dict[y][raw_data[x][y]];
            elif raw_data[x][y] == "nan":
                con_X[x][y] = np.nan;
            else:
                con_X[x][y] = np.float32(raw_data[x][y]);

    return feat_mask, feat_dict, con_X, con_Y;
    
def impute(raw_vals):
    imp = preprocessing.Imputer(missing_values='NaN', strategy='median', axis=0, verbose=0);
    imputed_vals = imp.fit_transform(raw_vals);
    return imputed_vals; 
 
def encode_cate_feats(raw_vals, feat_mask):
    enc = preprocessing.OneHotEncoder(n_values='auto', categorical_features=feat_mask);
    ext_vals = enc.fit_transform(raw_vals).toarray();
    return ext_vals;

def std_filter(ext_vals, th):
    col_stds = np.std(ext_vals, axis=0);
    std_vals = [ext_vals[:,i] for i in xrange(0, ext_vals.shape[1]) if col_stds[i] > th];
    return std_vals;
    
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



    