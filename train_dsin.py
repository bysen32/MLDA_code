# coding: utf-8
import os
import time
import argparse

import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras import backend as K

from config import DSIN_SESS_COUNT, DSIN_SESS_MAX_LEN, FRAC
from models import DSIN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
K.set_session(tf.Session(config=tfconfig))


def auroc(y_true,y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train dsin.')
    parser.add_argument('--frac', type=float, default=0, help='Fraction, 0 means using FRAC from config.')
    parser.add_argument('--epoch', type=int, default=1, help='Epoch, train how many times')
    parser.add_argument('--dataset', type=str, default=os.path.expanduser('~')+'/datasets/DSIN/', help='dataset path.')
    parser.add_argument('--batch', type=int, default=4096, help='batch size')
    parser.add_argument('--testbatch', type=int, default=2 ** 14, help='test batch size')
    parser.add_argument('--sub_index', type=int, default=0, help='sampled_data index.')

    args = parser.parse_args()
    print(args)

    if args.frac != 0:
        FRAC = args.frac
    EPOCH           = args.epoch
    BATCH_SIZE      = args.batch
    TEST_BATCH_SIZE = args.testbatch
    DATASET         = args.dataset

    SESS_COUNT = DSIN_SESS_COUNT
    SESS_MAX_LEN = DSIN_SESS_MAX_LEN

    sub = args.sub_index
    SAMP_PATH = DATASET + '/sampled_data/' + 'samp0' + str(sub) + '/'
    model_path = DATASET + '/model_input/samp0' + str(sub) + '_input/'

    fd          = pd.read_pickle(model_path + '/dsin_fd_' + str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    model_input = pd.read_pickle(model_path + '/dsin_input_' + str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    label       = pd.read_pickle(model_path + '/dsin_label_' + str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    sample_sub  = pd.read_pickle(SAMP_PATH  + '/raw_sample_' + str(FRAC) + '.pkl')
    print('load '+model_path + '/dsin_input_' + str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')

    sample_sub['idx'] = list(range(sample_sub.shape[0]))
    train_idx = sample_sub.loc[sample_sub.time_stamp < 1494633600, 'idx'].values
    test_idx = sample_sub.loc[sample_sub.time_stamp >= 1494633600, 'idx'].values

    train_input = [i[train_idx] for i in model_input]
    test_input = [i[test_idx] for i in model_input]

    train_label = label[train_idx]
    test_label = label[test_idx]

    sess_count = SESS_COUNT
    sess_len_max = SESS_MAX_LEN
    # BATCH_SIZE = 4096
    sess_feature = ['cate_id', 'brand']
    # TEST_BATCH_SIZE = 2 ** 14

    model = DSIN(fd, sess_feature, embedding_size=4, sess_max_count=sess_count,
                 sess_len_max=sess_len_max, dnn_hidden_units=(200, 80), att_head_num=8,
                 att_embedding_size=1, bias_encoding=False)

    model.compile('adagrad', 'binary_crossentropy',
                  metrics=['binary_crossentropy', auroc])

    hist_ = model.fit(train_input, train_label, batch_size=BATCH_SIZE,
                      epochs=EPOCH, initial_epoch=0, verbose=1, validation_data=(test_input, test_label))

    pred_ans = model.predict(test_input, TEST_BATCH_SIZE)

    print()
    print("test LogLoss", round(log_loss(test_label, pred_ans), 4), "test AUC",
          round(roc_auc_score(test_label, pred_ans), 4))

    pd.to_pickle(pred_ans, './pred_label_dsin_'+str(FRAC)+'.pkl')

