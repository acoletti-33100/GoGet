import os
import pandas as pd
from classification_extraction import train_valid_gg_cnn

"""
# Notes
    If you want to train the network and resume later, use:
        train_valid_gg_cnn(blacklist, batch_size, path_no_p,
                            path_10_p, path_6_9_p, checkpoint,
                            path_exp, True, False, checkpoint_path_save)
"""

df = pd.read_csv('config.csv', index_col=0, header=0, dtype=str)
batch_size = int(df.loc['batchSize'].values[0])
npy_dir = str(df.loc['pathH5s'].values[0])
path_exp_old = str(df.loc['pathSaveModel'].values[0])
path_no_p = npy_dir + os.sep + 'no_polyp.h5'
path_10_p = npy_dir + os.sep + 'polyp_10.h5'
path_6_9_p = npy_dir + os.sep + 'polyp_6_9.h5'
checkpoint_path = 'data' + os.sep + 'color' + os.sep + 'pos-neg1' + os.sep + 'train-weights' + os.sep + 'weights'
checkpoint = 'cp-{epoch:04d}.ckpt'
checkpoint = 'cp-0020.ckpt' + os.sep + 'variables' + os.sep + 'variables'
checkpoint = path_exp_old + os.sep + checkpoint
# path_exp_old = 'data' + os.sep + 'color' + os.sep + 'pos-neg'
# start training, too many images may not be possible
# so save checkpoint and resume later with the same weights
checkpoint_path_save = path_exp_old
blacklist = []
train_valid_gg_cnn(blacklist, batch_size, path_no_p,
                   path_10_p, path_6_9_p, checkpoint,
                   path_exp_old, False, False, checkpoint_path_save)
