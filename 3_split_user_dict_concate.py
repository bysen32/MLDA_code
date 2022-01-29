import pickle
import pandas as pd
import numpy as np
import argparse

# from pandas.core.groupby.generic import pin_whitelisted_properties
# import tqdm
from config import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data by user AND concate link list by user.')
    parser.add_argument('--frac', type=float, default=0, help='Fraction: 0 means using FRAC from config.')
    parser.add_argument('--piece', type=int, default=2, help='Piece: how many pieces the users are splited.')
    parser.add_argument('--fun', type=int, default=1, help='Function: 1 means split and 2 means concatenate')
    parser.add_argument('--link', type=int, default=5, help='Link: useful when concatenating')
    parser.add_argument('--feat', type=int, default=6, help='Link feature counts, 6:(id,pv,cart,fav,buy,gap), 8:(id,pv,cart,fav,buy,gap,begin,end).')
    parser.add_argument('--dataset', type=str, default='../', help='dataset path.')
    parser.add_argument('--sub_index', type=int, default=0, help='sampled_data index.')
    args = parser.parse_args()
    print (args)

    FEAT = args.feat
    piece = args.piece
    if args.frac != 0:
        FRAC = args.frac
    sub = args.sub_index
    DATASET = args.dataset
    if sub == -1:
        SAMP_PATH = DATASET + '/sampled_data/originsamp/'
    else:
        SAMP_PATH = DATASET + '/sampled_data/' + 'samp0' + str(sub) + '/'
    # split 
    if args.fun == 1:
        sample = pd.read_pickle( SAMP_PATH + '/raw_sample_' + str(FRAC) + '.pkl')
        sample.drop(columns=['adgroup_id','pid','nonclk','clk'], inplace=True)
        sample['idx'] = list(range(sample.shape[0]))
        sample_group = sample.groupby(['user'])
        user_sample_dict = {} 

        log = pd.read_pickle( SAMP_PATH + '/behavior_log_pv_user_filter_enc_btag_' + str(FRAC) + '.pkl')
        log = log.loc[log.time_stamp >= 1493769600]
        log_group = log.groupby(['user'])
        user_log_dict = {}
        
        # group to dict: sample and log
        for name, group in sample_group:
            user_sample_dict[name] = group
        for name, group in log_group:
            user_log_dict[name] = group
        
        userset = sample.user.unique()
        print('total user:',len(userset))
        
        if not os.path.exists(SAMP_PATH+'/sampled_user_info/'):
            os.mkdir(SAMP_PATH+'/sampled_user_info/')

        if piece == 0: # no split, just save full dict
            OUTPUT = SAMP_PATH + '/sampled_user_info/user_sample_log_dict_' + str(FRAC) + '.pkl'
            pd.to_pickle([userset, user_sample_dict, user_log_dict], OUTPUT)
            
        else: # split dict into pieces
            user_split = np.array_split(userset, piece)
            # sub dict
            count = 0
            for sub_users in user_split:
                user_sample_dict_sub = {key:value for key, value in user_sample_dict.items() if key in sub_users}
                user_log_dict_sub = {key:value for key, value in user_log_dict.items() if key in sub_users}
                OUTPUT = SAMP_PATH + '/sampled_user_info/user_sample_log_dict_' + str(FRAC) + '_piece_' + str(count) + '.pkl'
                pd.to_pickle([sub_users, user_sample_dict_sub, user_log_dict_sub], OUTPUT)

                count+=1
                print('sub_user', count, 'finish:', len(sub_users))
    # concate
    elif args.fun == 2:  
        LINK_MAX_INPUT = args.link
        link_path = DATASET + '/link/samp0' + str(sub) + '_link/' + 'fea_'+str(FEAT) + '/'

        user_link = []
        for i in range(piece):
            if not os.path.exists(link_path):
                os.mkdir(link_path)
            user_lk_ = pd.read_pickle( link_path +'/user_idx_clk_n_blk_n_'+str(FRAC)+'_lk'+str(LINK_MAX_INPUT)+'_fea'+str(FEAT)+'_sub'+str(i)+'.pkl')
            user_link.extend(user_lk_)
            del user_lk_

        pd.to_pickle(user_link, link_path + '/user_idx_clk_n_blk_n_'+str(FRAC)+'_lk'+str(LINK_MAX_INPUT)+'_fea'+str(FEAT)+'.pkl')
        print('save concatenated link file into', link_path+'/user_idx_clk_n_blk_n_'+str(FRAC)+'_lk'+str(LINK_MAX_INPUT)+'_fea'+str(FEAT)+'.pkl')
    





