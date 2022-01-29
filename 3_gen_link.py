import argparse
from datetime import time

from pandas.io import pickle
import tqdm
import joblib
import numpy as np
import pandas as pd
from multiprocessing import Pool
from config import *
from collections import Counter

pd.set_option('mode.chained_assignment', None)

# CHILD_LEN = 5 #?

def get_user_behaviour_list(uid, group, mode=0):
    behaviour_list = group.values[:, 2].tolist()
    # print(behaviour_list)
    deep_action = ['cart','fav','buy']

    if len(np.unique(behaviour_list)) >= 2 or (deep_action[0] in behaviour_list) or (deep_action[1] in behaviour_list) or (deep_action[2] in behaviour_list):
        if mode == 1:   # fea1
            return [uid]
        behaviour_count = str_to_num(behaviour_list)
        time_begin = group.values[:, 1].min()
        time_end   = group.values[:, 1].max()
        time_gap = time_end - time_begin
        if mode == 6: # no brand # fea_6
             return [uid, *behaviour_count, time_gap]
        else:
            MODE = mode
            # print('MODE=',mode)
            # select brands corresponding to deep action
            group_deep = group.loc[group['btag'].isin(deep_action)]
            cb_deep = np.unique(group_deep[MODE])
            len_deep = len(cb_deep)
            if len_deep < CHILD_LEN:
                # select most frequency brands
                len_rest = CHILD_LEN - len_deep
                pvlog = group.loc[group['btag'].isin(['pv'])]
                pvlog_nodeep = remove_list(pvlog[MODE], cb_deep) # remove cb_deep in pvlog
                pv_count = Counter(pvlog_nodeep)
                pv_count_sort = sorted(pv_count.items(), key=lambda x:x[1], reverse=True)
                len_freq = len(pv_count_sort)
                if len_freq >= len_rest:
                    cb_freq = [bi for bi,_ in pv_count_sort][0:len_rest]
                    child_id_list = [*cb_deep, *cb_freq]
                else:
                    cb_freq = [bi for bi,_ in pv_count_sort][0:len_freq]
                    child_id_list = [*cb_deep, *cb_freq, *[0]*(len_rest-len_freq)]
            else:
                child_id_list = cb_deep[0:CHILD_LEN]
            if FEAT==2:
                return [uid, *child_id_list]
            elif FEAT==11:
                return [uid, *child_id_list, *behaviour_count, time_gap]
    else:
        return None


def remove_list(iterable: list, target: list):
    return [n for n in iterable if n not in target]


def str_to_num(list):
    behaviour = ['pv','cart','fav','buy']
    return [list.count(b) for b in behaviour]


def np_unranked_unique(nparray):
    n_unique = len(np.unique(nparray))
    ranked_unique = [0]*n_unique
    i = 0
    for x in nparray:
        if x not in ranked_unique:
            ranked_unique[i] = x
            i += 1
    return ranked_unique


def gen_link_for_user(userid):
    sample_user = sample_user_dict[userid]
    user_link = []
    try:
        userlog = log_user_dict[userid]
    except:
        userlog = pd.DataFrame([])
    time_idx = sample_user.loc[:, ['time_stamp','idx']]
    for row in time_idx.iterrows():
        tst = row[1][0]
        idx = row[1][1]
        
        if userlog.shape[0] > 0:  
            user_log_tst = userlog[userlog.time_stamp < tst]     
            user_log_tst_desen = user_log_tst.sort_values("time_stamp",ascending=False,inplace=False) # Reverse
            # Reverse order id
            cate_recent = np_unranked_unique(user_log_tst_desen['cate'])
            brand_recent = np_unranked_unique(user_log_tst_desen['brand'])
            # group to dict
            user_cate_log_gr  = dict(list(user_log_tst_desen.groupby(['cate'])))
            user_brand_log_gr = dict(list(user_log_tst_desen.groupby(['brand'])))
            
            # generate link for cate
            link = []
            
            cmode = 1 if FEAT==1 else 6 if FEAT==6 else 'brand'
            for ci in cate_recent: # Reverse traversal
                link_ = get_user_behaviour_list(ci, user_cate_log_gr[ci], cmode)
                if link_ is not None:
                    link.append(link_)
                if len(link) == LINK_MAX_INPUT:
                    break
            clink_num = len(link)
            link.extend([[]]*(LINK_MAX_INPUT-clink_num))
            clink = link

            # generate link for brand
            link = []
            bmode = 1 if FEAT==1 else 6 if FEAT==6 else 'cate'
            for bi in brand_recent:
                link_ = get_user_behaviour_list(bi, user_brand_log_gr[bi], bmode)
                if link_ is not None:
                    link.append(link_)
                if len(link) == LINK_MAX_INPUT:
                    break
            blink_num = len(link)
            link.extend([[]]*(LINK_MAX_INPUT-blink_num))
            blink = link
            user_link.append((idx, clink, clink_num, blink, blink_num))
        else:
            user_link.append((idx, [], 0, [], 0))
    return user_link

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate link data.')
    parser.add_argument('--frac', type=float, default=0, help='fraction, 0 means using FRAC from config.')
    parser.add_argument('--userset_index', type=int, default=-1, help='userset index, -1 means no split.')
    parser.add_argument('--link', type=int, default=5, help='LINK_MAX_INPUT, default 5.')
    parser.add_argument('--child', type=int, default=5, help='one cate has how many child, default 5.')
    parser.add_argument('--feat', type=int, default=6, help='Link feature counts, 1:(only id), 6:(id,pv,cart,fav,buy,gap), 8:(id,pv,cart,fav,buy,gap,begin,end), 11:(id,child_list,pv,cart,fav,buy,gap)')
    parser.add_argument('--dataset', type=str, default='../', help='dataset path.')
    parser.add_argument('--sub_index', type=int, default=0, help='sampled_data index.')
    args = parser.parse_args()
    print (args)

    iu = args.userset_index
    LINK_MAX_INPUT = args.link
    CHILD_LEN = args.child
    FEAT = args.feat
    DATASET = args.dataset
    if args.frac != 0:
        FRAC = args.frac
    
    sub = args.sub_index
    if sub == -1:
        SAMP_PATH = DATASET + '/sampled_data/originsamp/'
        link_path = DATASET + '/link/origin/'
    else:
        SAMP_PATH = DATASET + '/sampled_data/' + 'samp0' + str(sub) + '/'
        link_path = DATASET + '/link/samp0' + str(sub) + '_link/' + 'fea_'+str(FEAT) + '/'

    # read pkl
    if args.userset_index == -1: # no split
        output = link_path + '/user_idx_clk_n_blk_n_' + str(FRAC) + '_lk' + str(LINK_MAX_INPUT) + '_fea'+str(FEAT) + '.pkl'
        userset, sample_user_dict, log_user_dict = pd.read_pickle( SAMP_PATH + '/sampled_user_info/user_sample_log_dict_' + str(FRAC) + '.pkl')
    else:
        output = link_path + '/user_idx_clk_n_blk_n_' + str(FRAC) + '_lk' + str(LINK_MAX_INPUT) + '_fea'+str(FEAT) + '_sub' + str(iu)  + '.pkl'
        userset, sample_user_dict, log_user_dict = pd.read_pickle( SAMP_PATH + '/sampled_user_info/user_sample_log_dict_' + str(FRAC) + '_piece_' + str(iu) +  '.pkl')

    user_num = len(userset)


    pool = Pool()
    res = list(tqdm.tqdm(pool.imap(gen_link_for_user, userset), total=user_num))
    with open(output,'wb') as f:
        joblib.dump(res, output)

    # joblib.dump(res,output)

    print('data saved to ' + output)
    # print('================================================================')

