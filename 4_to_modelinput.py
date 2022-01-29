
import joblib
import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import itertools

import argparse
from config import *


def standard_behavior(user_link, user_lk_count, sd_fun):
    USER_LINK = user_link  # 可变对象，这样还是会修改，copy.deepcopy()才可以
    behavior_count = 4
    sd_count = behavior_count + 1
    all_5num_list = [[]] * sd_count

    # 将USER_LINK的(-5:)列有效的行为数据+tst分别放在all_5num_list中
    for ni in range(sd_count):  # 0,1,2,3,4
        num_list = []
        for si in range(len(USER_LINK)):  # each sample idx
            count_val = user_lk_count[si]
            USER_LINK_val = USER_LINK[si][0:count_val]  # valid link
            num_list.append([ci[ni - 5] for ci in USER_LINK_val])  # 取-5 ~ -1列的数据

        all_5num_list[ni] = list(itertools.chain.from_iterable(num_list))  # 2dim list to 1dim

    # StandardScaler
    all_5num_list_t = np.array(all_5num_list, dtype='float32').T  # 每一列是同一个行为
    num_5fea_sds = np.zeros_like(all_5num_list_t, dtype='float32')

    if sd_fun == 'StandardScaler':
        for numi in range(sd_count):
            sds = StandardScaler()
            single_num = all_5num_list_t[:, numi].reshape(-1, 1)
            num_fea = list(itertools.chain.from_iterable(sds.fit_transform(single_num)))
            num_5fea_sds[:, numi] = num_fea
    elif sd_fun == 'MinMaxScaler':
        for numi in range(sd_count):
            mms = MinMaxScaler()
            single_num = all_5num_list_t[:, numi].reshape(-1, 1)
            num_fea = list(itertools.chain.from_iterable(mms.fit_transform(single_num)))
            num_5fea_sds[:, numi] = num_fea
    else:
        print('StandardScaler or MinMaxScaler')
	

    # 放入原来的 USER_LINK 中
    now_idx = 0
    for idx in range(0, len(USER_LINK)):  # sample
        count_val = user_lk_count[idx]  # link counts
        for ci in range(count_val):  # each link
            for cout in range(sd_count):  # each feat
                USER_LINK[idx][ci][cout - 5] = num_5fea_sds[:, cout][now_idx + ci]
        now_idx += count_val

    return USER_LINK


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate model input')
    parser.add_argument('--frac', type=float, default=0, help='Fraction, 0 means using FRAC from config.')
    parser.add_argument('--link', type=int, default=5, help='max_lk')
    parser.add_argument('--standard', type=int, default=0, help='Standard: 0 for no standard, 1 for StandardScaler, 2 for MinMaxScaler')
    parser.add_argument('--feat', type=int, default=6, help='Link feature counts, 1:(only id) 6:(id,pv,cart,fav,buy,gap), 8:(id,pv,cart,fav,buy,gap,begin,end).')
    parser.add_argument('--child', type=int, default=5, help='one cate has how many child, default 5.')
    parser.add_argument('--sub_index', type=int, default=0, help='sampled_data index.')
    parser.add_argument('--dataset', type=str, default='../', help='dataset path.')
    args = parser.parse_args()
    print(args)
    DATASET = args.dataset
    if args.frac != 0:
        FRAC = args.frac

    max_lk = args.link
    MAX_FEAT_LEN = args.feat
    CHILD_LEN = args.child

    sub = args.sub_index
    SAMP_PATH = DATASET + '/sampled_data/' + 'samp0' + str(sub) + '/'
    link_path = DATASET + '/link/samp0' + str(sub) + '_link/' + 'fea_'+str(MAX_FEAT_LEN) + '/'
    model_path = DATASET + '/model_input/samp0' + str(sub) + '_input/'

    if args.standard == 0:
        standard_fun = 'NoS'
    elif args.standard == 1:
        standard_fun = 'StandardScaler'
    elif args.standard == 2:
        standard_fun = 'MinMaxScaler'

    
    DSIN_model_input = pd.read_pickle(model_path + '/dsin_input_'+str(FRAC)+'_5.pkl')
    sample = pd.read_pickle( SAMP_PATH + '/raw_sample_' + str(FRAC) + '.pkl')
    SAMPLE_NUM = len(sample)
    print('Table sample has', SAMPLE_NUM,'lines')
    linkdata = pd.read_pickle(link_path + '/user_idx_clk_n_blk_n_'+str(FRAC)+'_lk'+str(max_lk)+'_fea'+str(MAX_FEAT_LEN)+'.pkl')
    del sample


    clink = [[] for _ in range(SAMPLE_NUM)]
    blink = [[] for _ in range(SAMPLE_NUM)]
    clk_input_count = [0 for _ in range(SAMPLE_NUM)]
    blk_input_count = [0 for _ in range(SAMPLE_NUM)]
    no_link = [[] for _ in range(max_lk)]  # padding no link to [[],[],[],[],[]]

    # read link from data
    print('Begin get link from data')
    for user in tqdm(linkdata):
        for cblink in user:
            idx = cblink[0]
            _, clink[idx], clk_input_count[idx], blink[idx], blk_input_count[idx] = cblink
            if clink[idx]==[]:
                clink[idx]=no_link
            if blink[idx]==[]:
                blink[idx]=no_link
    del linkdata


    if (MAX_FEAT_LEN == 6) or (MAX_FEAT_LEN == 11):# fea6 fea11 need standard, fea1 and fea2 do not need
        # =========================== standard ===========================
        if args.standard == 0:
            print('no stanard')
        else:
            clink = standard_behavior(clink, clk_input_count, standard_fun)
            blink = standard_behavior(blink, blk_input_count, standard_fun)
            print(standard_fun, 'success')
        # ===========================
    
    clink_pad = []
    blink_pad = []
    # padding [] to [0,0,0,...,0,0] cid,pv,cart,fav,buy,gap
    print('Begin padding')
    pad_len = 1+CHILD_LEN if MAX_FEAT_LEN==2 else MAX_FEAT_LEN
    for i in tqdm(range(SAMPLE_NUM)):
        clink_pad.append(pad_sequences(clink[i], maxlen=pad_len, padding='post').tolist())
        blink_pad.append(pad_sequences(blink[i], maxlen=pad_len, padding='post').tolist())
    del clink, blink

    cate_input =[]
    brand_input=[]
    # 用户的5个 link 分别放在 cate_0 ~ cate_4
    print('Begin reorganizing')
    for j in tqdm(range(max_lk)):
        brand_input.append(np.array([[blink_pad[bi][j]] for bi in range(len(blk_input_count))]))
        cate_input.append(np.array([[clink_pad[ci][j]] for ci in range(len(clk_input_count))]))
    del blink_pad, clink_pad
    

    # 27+5+5+1+1=39
    new_model_input= DSIN_model_input + brand_input + cate_input +\
                    [np.array(blk_input_count)]+\
                    [np.array(clk_input_count)]

    datapath = model_path + '/input_' + str(FRAC) + '_lk' + str(max_lk) + '_fea' + str(MAX_FEAT_LEN) + '_' + standard_fun + '.pkl'
    pd.to_pickle(new_model_input, datapath)
    print('save new model_input to', datapath)
