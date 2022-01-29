# coding: utf-8
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import *

def chunk_read_log(user_sub):
    userset = user_sub.userid.unique()
    filepath= open( DATASET+ "/raw_data/behavior_log.csv",errors="ignore")  # 指定文件路径
    reader = pd.read_csv(filepath,
                        header=None,
                        names=["user","time_stamp","btag","cate","brand"],  # 指定列属性名称
                        iterator=True)

    # loop,chunkSize,chunks = True, 10000000, []  # 连续赋值语句
    loop = True
    chunkSize = 10000000
    chunks = []
    i=0

    while loop:  # loop一直为True，执行循环
        try:
            chunk = reader.get_chunk(chunkSize)
            ###### chunk = chunk.loc[chunk['btag'] == 'pv']  # 只分析 pv 的历史数据，（浏览的）
            sampled_chunk = chunk.loc[chunk.user.isin(userset)]
            chunks.append(sampled_chunk)
            print (i)
            i = i+1
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    # print("Finished read log!")
    df = pd.concat(chunks, ignore_index=True)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate link data.')
    parser.add_argument('--frac', type=float, default=0, help='fraction, 0 means using FRAC from config.')
    # parser.add_argument('--dataset', type=str, default='../', help='dataset path.')
    args = parser.parse_args()
    print (args)

    if args.frac != 0:
        FRAC = args.frac

    print('FRAC=',FRAC)
    user = pd.read_csv( DATASET+ '/raw_data/user_profile.csv')
    sample = pd.read_csv( DATASET+ '/raw_data/raw_sample.csv')

    if not os.path.exists( DATASET+ '/sampled_data/'):
        os.mkdir( DATASET+ '/sampled_data/')


    if os.path.exists( DATASET+ '/sampled_data/user_profile_' + str(FRAC) + '.pkl') and os.path.exists(
             DATASET+ '/sampled_data/raw_sample_' + str(FRAC) + '.pkl'):
        user_sub = pd.read_pickle(
             DATASET+ '/sampled_data/user_profile_' + str(FRAC) + '.pkl')
        sample_sub = pd.read_pickle(
             DATASET+ '/sampled_data/raw_sample_' + str(FRAC) + '.pkl')
    else:

        if FRAC < 1.0:
            user_sub = user.sample(frac=FRAC, random_state=1024) #随机采的用户
        else:
            user_sub = user
        sample_sub = sample.loc[sample.user.isin(user_sub.userid.unique())]
        pd.to_pickle(user_sub,  DATASET+ '/sampled_data/user_profile_' +
                     str(FRAC) + '.pkl')
        pd.to_pickle(sample_sub,  DATASET+ '/sampled_data/raw_sample_' +
                     str(FRAC) + '.pkl')

    if os.path.exists( DATASET+ '/raw_data/behavior_log_pv.pkl'):
        log = pd.read_pickle( DATASET+ '/raw_data/behavior_log_pv.pkl')
    else:
        log = chunk_read_log(user_sub)

    ad = pd.read_csv( DATASET+ '/raw_data/ad_feature.csv')
    ad['brand'] = ad['brand'].fillna(-1)

    lbe = LabelEncoder()
    # unique_cate_id = ad['cate_id'].unique()
    # log = log.loc[log.cate.isin(unique_cate_id)]

    unique_cate_id = np.concatenate(
        (ad['cate_id'].unique(), log['cate'].unique()))

    lbe.fit(unique_cate_id)             ## 训练 LabelEncoder 将 unique_cate_id 中数据编码
    ad['cate_id'] = lbe.transform(ad['cate_id']) + 1    ##将训练好的 LabelEncoder 对原始数据进行编码，从 1 开始
    log['cate'] = lbe.transform(log['cate']) + 1

    lbe = LabelEncoder()
    # unique_brand = np.ad['brand'].unique()
    # log = log.loc[log.brand.isin(unique_brand)]

    unique_brand = np.concatenate(
        (ad['brand'].unique(), log['brand'].unique()))

    lbe.fit(unique_brand)
    ad['brand'] = lbe.transform(ad['brand']) + 1
    log['brand'] = lbe.transform(log['brand']) + 1

    log = log.loc[log.user.isin(sample_sub.user.unique())]
    ###### log.drop(columns=['btag'], inplace=True)    # 去掉 btag 列，直接在原始数据上drop，返回空。 inplace=false返回copy（默认）
    log = log.loc[log['time_stamp'] > 0]

    pd.to_pickle(ad,  DATASET+ '/sampled_data/ad_feature_enc_' + str(FRAC) + '.pkl')
    pd.to_pickle(
        log,  DATASET+ '/sampled_data/behavior_log_pv_user_filter_enc_btag_' + str(FRAC) + '.pkl')

    print("0_gen_sampled_data_btag done")
