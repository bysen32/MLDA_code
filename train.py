import os
import time
import argparse
import random
import numpy as np

import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard
from config import *
from models import DSINLK1
from models import DSINLK6
from models import DSINCHILD2
from models import DSINCHILD11


SESS_COUNT = DSIN_SESS_COUNT
SESS_MAX_LEN = DSIN_SESS_MAX_LEN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.7
K.set_session(tf.Session(config=tfconfig))


def seed_tensorflow(seed=42):
    print('seed=',seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


if __name__ == "__main__":
    # seed_tensorflow(42)
    # arguments
    parser = argparse.ArgumentParser(description='Train dsinlink file')
    parser.add_argument('--frac', type=float, default=0, help='Fraction, 0 means using FRAC from config.')
    parser.add_argument('--link', type=int, default=5, help='max_lk')
    parser.add_argument('--epoch', type=int, default=1, help='Epoch, train how many times')
    parser.add_argument('--standard', type=int, default=0, help='Standard: 0 for no standard, 1 for StandardScaler, 2 for MinMaxScaler')
    parser.add_argument('--feat', type=int, default=6, help='Link feature counts:{1,2,6,11}, 6:(id,pv,cart,fav,buy,gap), 8:(id,pv,cart,fav,buy,gap,begin,end).')
    # parser.add_argument('--model', type=int, default=2, help='model_type: 2: DSINLK, 3: DSINCHILD')
    parser.add_argument('--dataset', type=str, default=os.path.expanduser('~')+'/datasets/DSIN/', help='dataset path.')
    parser.add_argument('--batch', type=int, default=4096, help='batch size')
    parser.add_argument('--testbatch', type=int, default=2 ** 14, help='test batch size')
    parser.add_argument('--sub_index', type=int, default=0, help='sampled_data index.')

    args = parser.parse_args()
    print(args)
    # arguments assignment
    if args.frac != 0:
        FRAC = args.frac
    max_lk          = args.link
    EPOCH           = args.epoch
    MAX_FEAT_LEN    = args.feat
    # model_type      = args.model
    BATCH_SIZE      = args.batch
    TEST_BATCH_SIZE = args.testbatch
    sdd             = args.standard
    DATASET         = args.dataset
    standard = 'StandardScaler' if sdd==1 else 'MinMaxScaler' if sdd==2 else 'NoS'

    sub = args.sub_index
    SAMP_PATH = DATASET + '/sampled_data/' + 'samp0' + str(sub) + '/'
    model_path = DATASET + '/model_input/samp0' + str(sub) + '_input/'
    # load file
    fd          = pd.read_pickle( model_path +'/dsin_fd_'+str(FRAC)+'_'+str(SESS_COUNT)+'.pkl')
    model_input = pd.read_pickle( model_path +'/input_'+str(FRAC)+'_lk'+str(max_lk)+'_fea'+str(MAX_FEAT_LEN)+'_'+standard+'.pkl')
    label       = pd.read_pickle( model_path +'/dsin_label_'+str(FRAC)+'_'+str(SESS_COUNT)+'.pkl')
    sample_sub  = pd.read_pickle( SAMP_PATH  +'/raw_sample_'+str(FRAC) +'.pkl')
    print('load '+model_path+'/input_'+str(FRAC)+'_lk'+str(max_lk)+'_fea'+str(MAX_FEAT_LEN)+'_'+standard+'.pkl')

    # divide train set
    sample_sub['idx'] = list(range(sample_sub.shape[0]))
    train_idx = sample_sub.loc[sample_sub.time_stamp < 1494633600, 'idx'].values
    test_idx  = sample_sub.loc[sample_sub.time_stamp >= 1494633600, 'idx'].values
    train_input = [i[train_idx] for i in model_input]
    test_input  = [i[test_idx] for i in model_input]
    train_label = label[train_idx]
    test_label  = label[test_idx]

    sess_count = SESS_COUNT
    sess_len_max = SESS_MAX_LEN
    sess_feature = ['cate_id', 'brand']
    

    # model_v2
    if MAX_FEAT_LEN == 1:
        print('model:DSINLK1')
        model = DSINLK1(fd, sess_feature, embedding_size=4, sess_max_count=sess_count,
                        sess_len_max=sess_len_max, dnn_hidden_units=(200, 80), att_head_num=8,
                        att_embedding_size=1, bias_encoding=False, max_lk_len=max_lk, emb_len=MAX_FEAT_LEN)
    elif MAX_FEAT_LEN == 6:
        print('model:DSINLK6')
        model = DSINLK6(fd, sess_feature, embedding_size=4, sess_max_count=sess_count,
            sess_len_max=sess_len_max, dnn_hidden_units=(200, 80), att_head_num=8,
            att_embedding_size=1, bias_encoding=False, max_lk_len=max_lk, emb_len=MAX_FEAT_LEN)
    # model_v3
    elif MAX_FEAT_LEN == 2:
        print('model:DSINCHILD2')
        model = DSINCHILD2(fd, sess_feature, embedding_size=4, sess_max_count=sess_count,
                sess_len_max=sess_len_max, dnn_hidden_units=(200, 80), att_head_num=8,
                att_embedding_size=1, bias_encoding=False, max_lk_len=max_lk, emb_len=MAX_FEAT_LEN)
    elif MAX_FEAT_LEN == 11:
        print('model:DSINCHILD11')
        model = DSINCHILD11(fd, sess_feature, embedding_size=4, sess_max_count=sess_count,
                sess_len_max=sess_len_max, dnn_hidden_units=(200, 80), att_head_num=8,
                att_embedding_size=1, bias_encoding=False, max_lk_len=max_lk, emb_len=MAX_FEAT_LEN)
    else:
        print("Feat should in {1,2,6,11}")
    model.compile('adagrad', 'binary_crossentropy',
                  metrics=['binary_crossentropy', auroc])
    # log_dir=".logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    hist = model.fit(train_input, train_label, batch_size=BATCH_SIZE,
                      epochs=EPOCH, initial_epoch=0, verbose=2, validation_data=(test_input, test_label))#callbacks=[LossHistory(log_dir)]) #callbacks=[batch_print_callback]) # 
    # print(hist.history)
    pred_ans = model.predict(test_input, TEST_BATCH_SIZE)


    print("test LogLoss", round(log_loss(test_label, pred_ans), 4), "test AUC",
          round(roc_auc_score(test_label, pred_ans), 4))
    print("")

    # time_e = time.time()


    pd.to_pickle(pred_ans, './pred_label_model2_'+str(FRAC)+'.pkl')




    with open("results.txt","a") as f:
        f.write(str(args))
        f.write("\ntestLogLoss=%f"%(round(log_loss(test_label, pred_ans), 4))+", testAUC=%f\n\n"%(round(roc_auc_score(test_label, pred_ans), 4)))
    f.close()
    K.clear_session()