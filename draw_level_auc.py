import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score



def gen_test_cate_auc(pred_ans, MODE, train_clkcount):
    # get test sample
    test_idx = sample_aid.loc[sample_aid.time_stamp >= 1494633600, 'idx'].values
    test_label = label[test_idx]
    test_sample = sample_aid.loc[sample_aid['idx'].isin(list(test_idx))].copy()
    test_sample['label']=test_label
    test_sample['pre_ans']=pred_ans

    # group
    test_sample_gr = test_sample.groupby([MODE])

    # select id who has 01 label
    lb01_cbid_list=[]
    for cid,cgr in test_sample_gr:
        if len(np.unique(list(cgr.label)))>=2:
            lb01_cbid_list.append(cid)

    lb01_count = len(lb01_cbid_list)
    test_cid_cout = len(test_sample_gr)
    print(round(lb01_count/test_cid_cout,4),' test samples have pos&neg label.('+str(lb01_count)+'/'+str(test_cid_cout))
    


    # get the pos train sample count of 01-label test id
    traincount_of_testid={}
    for cid in lb01_cbid_list:
        if cid in train_clkcount.keys():
            traincount_of_testid[cid]=train_clkcount[cid]
        else:
            traincount_of_testid[cid]=0

    # cal auc
    lb01_cbid_list_auc={}
    for cli in lb01_cbid_list:
        a_lb = test_sample.loc[test_sample[MODE] == cli,'label'].values
        a_pr = test_sample.loc[test_sample[MODE] == cli,'pre_ans'].values
        lb01_cbid_list_auc[cli]=roc_auc_score(a_lb, a_pr)

    return list(traincount_of_testid.values()), list(lb01_cbid_list_auc.values())


if __name__ == "__main__":
    DATASET='../'
    sub=0
    FRAC=0.001
    SESS_COUNT=5

    SAMP_PATH = DATASET + '/sampled_data/' + 'samp0' + str(sub) + '/'
    model_path = DATASET + '/model_input/samp0' + str(sub) + '_input/'


    label       = pd.read_pickle(model_path + '/dsin_label_' + str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    sample_sub  = pd.read_pickle(SAMP_PATH  + '/raw_sample_' + str(FRAC) + '.pkl')
    ad          = pd.read_pickle(SAMP_PATH  + 'ad_feature_enc_'+ str(FRAC) + '.pkl')

    dsin_pred_ans, dsin_test_label, dsin_train_idx, dsin_test_idx = pd.read_pickle('pred_label_dsin_'+str(FRAC)+'.pkl')
    our_pred_ans, our_test_label, our_train_idx, our_test_idx = pd.read_pickle('pred_label_model2_'+str(FRAC)+'.pkl')


    sample_sub['idx'] = list(range(sample_sub.shape[0]))
    sample_aid =sample_sub[['time_stamp','adgroup_id','clk','idx']].copy()
    sample_aid['cate_id']=list([0 for i in range(sample_sub.shape[0])])
    sample_aid['brand']=list([0 for i in range(sample_sub.shape[0])])

    # find and insert cate_id and brand
    unique_aid = set(sample_aid['adgroup_id'])
    for ai in unique_aid:
        cor_cid = int(ad.loc[ad['adgroup_id']==ai,'cate_id'])
        cor_bid = int(ad.loc[ad['adgroup_id']==ai,'brand'])
        sample_aid.loc[sample_aid.adgroup_id==ai,'cate_id'] = cor_cid
        sample_aid.loc[sample_aid.adgroup_id==ai,'brand'] = cor_bid
        
    # get train sample
    train_idx = sample_aid.loc[sample_aid.time_stamp < 1494633600, 'idx'].values
    train_label = label[train_idx]
    train_sample = sample_aid.loc[sample_aid['idx'].isin(list(train_idx))].copy()
    # cal pos sample count in train sample
    train_sample_grcid = train_sample.groupby(['cate_id'])
    train_sample_grbid = train_sample.groupby(['brand'])
    cate_clkcount={}
    brand_clkcount={}
    for cid,cgr in train_sample_grcid: # cate
        cate_clkcount[cid]=sum(cgr.clk==1)
    for bid,bgr in train_sample_grbid: #brand
        brand_clkcount[bid]=sum(bgr.clk==1)

    sample_gr_count = len(cate_clkcount)
    sample_pos_count = len([i for i in list(cate_clkcount.values()) if i!=0])
    print(round(sample_pos_count/sample_gr_count, 4), 'positave training cate samples.('+str(sample_pos_count)+'/'+str(sample_count))


    cal_mode = 'cate_id'
    x1,y1 = gen_test_cate_auc(our_pred_ans ,'cate_id', cate_clkcount)
    x2,y2 = gen_test_cate_auc(dsin_pred_ans,'cate_id', cate_clkcount)
    # x1,y1 = gen_test_cate_auc(our_pred_ans ,'brand', brand_clkcount)
    # x2,y2 = gen_test_cate_auc(dsin_pred_ans,'brand', brand_clkcount)

    if list(x1)==list(x2):
        plt.xlabel('Occurrences of positive sample')
        plt.ylabel('AUC')
        plt.title(cal_mode+': FRAC='+str(FRAC)+', SUB='+str(sub))

        plt.scatter(x1, y1, label='Ours')
        plt.scatter(x2, y2, label='DSIN')
        plt.legend(loc='upper right')
        
        # gcf: Get Current Figure
        fig = plt.gcf()
        plt.show()
        # fig.savefig(cal_mode+'_FRAC'+str(FRAC)+'_SUB'+str(sub)+'.pdf',dpi=600)
    else:
        print('x1 != x2')