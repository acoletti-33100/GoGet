import os
import pandas as pd
from classification_extraction import classification, features_extraction
from utils import from_str_to_bool

"""
# Notes
    the bigger the h5 datasets are, the more time it will take for feature extraction.
"""

blacklist = []
df = pd.read_csv('config.csv', index_col=0, header=0, dtype=str)
path_xls_no_polyp = str(df.loc['pathXlsNoPolyp'].values[0])
path_xls_10_polyp = str(df.loc['pathXls10Polyp'].values[0])
path_xls_6_9_polyp = str(df.loc['pathXls69Polyp'].values[0])
batch_size = int(df.loc['batchSize'].values[0])
path_pos_ftrs = str(df.loc['pathPosFtrs'].values[0])
path_neg_ftrs = str(df.loc['pathNegFtrs'].values[0])
pngs = str(df.loc['pathPngs'].values[0])
npy_dir = str(df.loc['pathH5s'].values[0])
path_no_p = npy_dir + os.sep + 'no_polyp.h5'
path_10_p = npy_dir + os.sep + 'polyp_10.h5'
path_6_9_p = npy_dir + os.sep + 'polyp_6_9.h5'
path_weights = str(df.loc['pathWeights'].values[0])
do_feature_extraction = str(df.loc['doFeatureExtraction'].values[0])
do_feature_extraction = from_str_to_bool(do_feature_extraction)
do_classification = str(df.loc['doClassification'].values[0])
do_classification = from_str_to_bool(do_classification)
if do_feature_extraction:
    features_extraction(blacklist, path_weights, pngs,
                        batch_size, path_pos_ftrs, path_neg_ftrs,
                        path_xls_no_polyp, path_xls_10_polyp, path_xls_6_9_polyp,
                        path_no_p, path_10_p, path_6_9_p)
path_pos_class = path_pos_ftrs + os.sep + 'features'
path_neg_class = path_neg_ftrs + os.sep + 'features'
path_images = str(df.loc['pathSaveImages'].values[0])
if do_classification:
    classification(path_pos_class, path_neg_class, path_images)
