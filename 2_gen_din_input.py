# coding: utf-8

import os
import argparse
import numpy as np
import pandas as pd
from deepctr.utils import SingleFeat
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from config import DIN_SESS_MAX_LEN, FRAC, DATASET
FRAC=0.25
print(FRAC)
def gen_sess_feature_din(row):
    sess_max_len = DIN_SESS_MAX_LEN
    sess_input_dict = {'cate_id': [0], 'brand': [0]}
    sess_input_length = 0
    user, time_stamp = row[1]['user'], row[1]['time_stamp']
    if user not in user_hist_session or len(user_hist_session[user]) == 0:

        sess_input_dict['cate_id'] = [0]
        sess_input_dict['brand'] = [0]
        sess_input_length = 0
    else:
        cur_sess = user_hist_session[user][0]
        for i in reversed(range(len(cur_sess))):
            if cur_sess[i][2] < time_stamp:
                sess_input_dict['cate_id'] = [e[0]
                                              for e in cur_sess[max(0, i + 1 - sess_max_len):i + 1]]
                sess_input_dict['brand'] = [e[1]
                                            for e in cur_sess[max(0, i + 1 - sess_max_len):i + 1]]
                sess_input_length = len(sess_input_dict['brand'])
                break
    return sess_input_dict['cate_id'], sess_input_dict['brand'], sess_input_length


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate session.')
    parser.add_argument('--frac', type=float, default=0, help='fraction, 0 means using FRAC from config.')
    parser.add_argument('--dataset', type=str, default='../', help='dataset path.')
    parser.add_argument('--sub_index', type=int, default=0, help='sampled_data index.')

    args = parser.parse_args()
    print (args)
    if args.frac != 0:
        FRAC = args.frac
    DATASET = args.dataset
    sub = args.sub_index

    SAMP_PATH = DATASET + '/sampled_data/' + 'samp0' + str(sub) + '/'
    model_path = DATASET + '/model_input/samp0' + str(sub) + '_input/'

    user_hist_session = {}
    FILE_NUM = len(
        list(
            filter(lambda x: x.startswith('user_hist_session_' + str(FRAC) + '_din_'), os.listdir(SAMP_PATH))))

    print('total', FILE_NUM, 'files')
    for i in range(FILE_NUM):
        user_hist_session_ = pd.read_pickle(
            SAMP_PATH + '/user_hist_session_' + str(FRAC) + '_din_' + str(i) + '.pkl')
        user_hist_session.update(user_hist_session_)
        del user_hist_session_

    sample_sub = pd.read_pickle(
        SAMP_PATH + '/raw_sample_' + str(FRAC) + '.pkl')

    sess_input_dict = {'cate_id': [], 'brand': []}
    sess_input_length = []
    for row in tqdm(sample_sub[['user', 'time_stamp']].iterrows()):
        a, b, c = gen_sess_feature_din(row)
        sess_input_dict['cate_id'].append(a)
        sess_input_dict['brand'].append(b)
        sess_input_length.append(c)

    print('done')

    user = pd.read_pickle( SAMP_PATH + '/user_profile_' + str(FRAC) + '.pkl')
    ad = pd.read_pickle( SAMP_PATH + '/ad_feature_enc_' + str(FRAC) + '.pkl')
    user = user.fillna(-1)
    user.rename(
        columns={'new_user_class_level ': 'new_user_class_level'}, inplace=True)

    sample_sub = pd.read_pickle(
         SAMP_PATH + '/raw_sample_' + str(FRAC) + '.pkl')
    sample_sub.rename(columns={'user': 'userid'}, inplace=True)

    data = pd.merge(sample_sub, user, how='left', on='userid', )
    data = pd.merge(data, ad, how='left', on='adgroup_id')

    sparse_features = ['userid', 'adgroup_id', 'pid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
                       'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'campaign_id',
                       'customer']
    dense_features = ['price']

    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()  # or Hash
        data[feat] = lbe.fit_transform(data[feat])
    mms = StandardScaler()
    data[dense_features] = mms.fit_transform(data[dense_features])

    sparse_feature_list = [SingleFeat(feat, data[feat].max(
    ) + 1) for feat in sparse_features + ['cate_id', 'brand']]

    dense_feature_list = [SingleFeat(feat, 1) for feat in dense_features]
    sess_feature = ['cate_id', 'brand']

    sess_input = [pad_sequences(
        sess_input_dict[feat], maxlen=DIN_SESS_MAX_LEN, padding='post') for feat in sess_feature]

    model_input = [data[feat.name].values for feat in sparse_feature_list] + \
                  [data[feat.name].values for feat in dense_feature_list]
    sess_lists = sess_input  # + [np.array(sess_input_length)]
    model_input += sess_lists

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    pd.to_pickle(model_input, model_path + '/din_input_' +
                 str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl')
    pd.to_pickle([np.array(sess_input_length)], model_path + '/din_input_len_' +
                 str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl')

    pd.to_pickle(data['clk'].values, model_path + '/din_label_' +
                 str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl')
    pd.to_pickle({'sparse': sparse_feature_list, 'dense': dense_feature_list},
                 model_path + '/din_fd_' + str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl', )

    print("gen din input of samp0" + str(sub) + " done")
