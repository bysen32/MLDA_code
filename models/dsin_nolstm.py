# coding: utf-8
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Feng Y, Lv F, Shen W, et al. Deep Session Interest Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1905.06482, 2019.(https://arxiv.org/abs/1905.06482)

"""

from collections import OrderedDict

from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import (Concatenate, Dense, Embedding,
                                            Flatten, Input)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2

from deepctr.input_embedding import (create_singlefeat_inputdict,
                                     get_embedding_vec_list, get_inputs_list)
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.sequence import (AttentionSequencePoolingLayer, BiasEncoding,
                                     BiLSTM, Transformer)
from deepctr.layers.utils import NoMask, concat_fun
from deepctr.utils import check_feature_config_dict

def DSINNL(feature_dim_dict, sess_feature_list, embedding_size=8, sess_max_count=5, sess_len_max=10,
         bias_encoding=False,
         att_embedding_size=1, att_head_num=8, dnn_hidden_units=(200, 80), dnn_activation='sigmoid', dnn_dropout=0,
         dnn_use_bn=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, init_std=0.0001, seed=1024, task='binary',
         ):
    """Instantiates the Deep Session Interest Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field (**now only support sparse feature**)like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':[]}
    :param sess_feature_list: list,to indicate session feature sparse field (**now only support sparse feature**),must be a subset of ``feature_dim_dict["sparse"]``
    :param embedding_size: positive integer,sparse feature embedding_size.
    :param sess_max_count: positive int, to indicate the max number of sessions
    :param sess_len_max: positive int, to indicate the max length of each session
    :param bias_encoding: bool. Whether use bias encoding or postional encoding
    :param att_embedding_size: positive int, the embedding size of each attention head
    :param att_head_num: positive int, the number of attention head
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """
    print('DSIN without lstm and corresponding emb')
    check_feature_config_dict(feature_dim_dict)

    if (att_embedding_size * att_head_num != len(sess_feature_list) * embedding_size):  # 8=2*4
        raise ValueError(
            "len(session_feature_lsit) * embedding_size must equal to att_embedding_size * att_head_num ,got %d * %d != %d *%d" % (
                len(sess_feature_list), embedding_size, att_embedding_size, att_head_num))

    sparse_input, dense_input, user_behavior_input_dict, _, user_sess_length = get_input(  # input layer
        feature_dim_dict, sess_feature_list, sess_max_count, sess_len_max)

    # 15 ??? embedding
    sparse_embedding_dict = {feat.name: Embedding(feat.dimension,  # ???????????????????????????????????????
                                                  embedding_size,  # 4
                                                  embeddings_initializer=RandomNormal(  # ?????????????????????
                                                      mean=0.0, stddev=init_std, seed=seed),
                                                  embeddings_regularizer=l2(  # ????????????????????????
                                                      l2_reg_embedding),
                                                  name='sparse_emb_' +
                                                       str(i) + '-' + feat.name,
                                                  mask_zero=(feat.name in sess_feature_list)) for i, feat in
                             enumerate(
                                 feature_dim_dict["sparse"])}  # mask_zero = Ture ?????? cate_id ??? brand ?????? embedding ??????

    # === cate_id, brand ??? Tensor???[<tf.Tensor shape=(?, 1, 4) >,<tf.Tensor shape=(?, 1, 4)>]
    query_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict["sparse"],
                                            sess_feature_list, sess_feature_list)
    query_emb = concat_fun(query_emb_list)  # <tf.Tensor 'concatenate/concat:0' shape=(?, 1, 8) dtype=float32>

    # === 15 ??? sparse feat ??? Tensor???[<tf.Tensor shape=(?, 1, 4)>,<>,...,<>]
    deep_input_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict["sparse"],
                                                 mask_feat_list=sess_feature_list)  # no return_feat_list means all
    deep_input_emb = concat_fun(
        deep_input_emb_list)  # <tf.Tensor 'concatenate_2/concat:0' shape=(?, 1, 60) dtype=float32>
    deep_input_emb = Flatten()(NoMask()(
        deep_input_emb))  # Flatten???????????????  # NoMask? # <tf.Tensor 'flatten/Reshape:0' shape=(?, 60) dtype=float32>

    # 5 ??? sess ???  (None,10,8)  cate_id+brand Tensor,    [<tf.Tensor shape=(?, 10, 8) >, <>, <>, <>, <>]  10=5*2
    tr_input = sess_interest_division(sparse_embedding_dict, user_behavior_input_dict,
                                      feature_dim_dict['sparse'], sess_feature_list, sess_max_count,
                                      bias_encoding=bias_encoding)

    #                                    1                 8
    Self_Attention = Transformer(att_embedding_size, att_head_num, dropout_rate=0, use_layer_norm=False,
                                 use_positional_encoding=(not bias_encoding), seed=seed, supports_masking=True,
                                 blinding=True)

    # I_k
    # sess_fea: <tf.Tensor 'concatenate_8/concat:0' shape=(?, 5, 8) dtype=float32>
    sess_fea = sess_interest_extractor(
        tr_input, sess_max_count, Self_Attention)

    # <tf.Tensor 'attention_sequence_pooling_layer/MatMul:0' shape=(?, 1, 8) dtype=float32>
    interest_attention_layer = AttentionSequencePoolingLayer(att_hidden_units=(64, 16), weight_normalization=True,
                                                             supports_masking=False)(
        [query_emb, sess_fea, user_sess_length])

    # H_k
    # # lstm_outputs: Tensor("bi_lstm/truediv:0", shape=(?, 5, 8), dtype=float32)
    # lstm_outputs = BiLSTM(len(sess_feature_list) * embedding_size,
    #                       layers=2, res_layers=0, dropout_rate=0.2, )(sess_fea)
    # # lstm_attention_layer: Tensor("attention_sequence_pooling_layer_2/MatMul:0", shape=(?, 1, 8), dtype=float32)
    # lstm_attention_layer = AttentionSequencePoolingLayer(att_hidden_units=(64, 16), weight_normalization=True)(
    #     [query_emb, lstm_outputs, user_sess_length])

    deep_input_emb = Concatenate()(
        [deep_input_emb, Flatten()(interest_attention_layer) #, Flatten()(lstm_attention_layer),
         ])  # 60+8=68
    if len(dense_input) > 0:
        deep_input_emb = Concatenate()(
            [deep_input_emb] + list(dense_input.values()))  # 69

    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                 dnn_dropout, dnn_use_bn, seed)(deep_input_emb)
    output = Dense(1, use_bias=False, activation=None)(output)  # ????????????
    output = PredictionLayer(task)(output)  # binary ?????????

    sess_input_list = []
    # sess_input_length_list = []
    for i in range(sess_max_count):
        sess_name = "sess_" + str(i)
        sess_input_list.extend(get_inputs_list(
            [user_behavior_input_dict[sess_name]]))
        # sess_input_length_list.append(user_behavior_length_dict[sess_name])

    model_input_list = get_inputs_list([sparse_input, dense_input]) + sess_input_list + [
        user_sess_length]

    model = Model(inputs=model_input_list, outputs=output)  # ????????????????????????

    return model

##############|      fd,      | ['cate_id', 'brand']|     5      |     10     |
def get_input(feature_dim_dict, seq_feature_list, sess_max_count, seq_max_len):
    sparse_input, dense_input = create_singlefeat_inputdict(feature_dim_dict)  # ?????????????????????
    user_behavior_input = {}
    for idx in range(sess_max_count):
        sess_input = OrderedDict()
        for i, feat in enumerate(seq_feature_list):
            sess_input[feat] = Input(
                shape=(seq_max_len,), name='seq_' + str(idx) + str(i) + '-' + feat)  # seq_00-cate_id, seq_01-brand

        user_behavior_input["sess_" + str(idx)] = sess_input

    user_behavior_length = {"sess_" + str(idx): Input(shape=(1,), name='seq_length' + str(idx)) for idx in
                            range(sess_max_count)}
    user_sess_length = Input(shape=(1,), name='sess_length')

    return sparse_input, dense_input, user_behavior_input, user_behavior_length, user_sess_length


def sess_interest_division(sparse_embedding_dict,  # Embedding 4 dim
                           user_behavior_input_dict,
                           # {'sess_0': OrderedDict([(..seq_00-cate_id),(..seq_01-brand)]),...'':}
                           sparse_fg_list,  # fd['sparse']=[SingleFeat(),...,]
                           sess_feture_list,  # ['cate_id', 'brand']
                           sess_max_count,  # 5
                           bias_encoding=True):
    tr_input = []
    for i in range(sess_max_count):
        sess_name = "sess_" + str(i)
        keys_emb_list = get_embedding_vec_list(sparse_embedding_dict, user_behavior_input_dict[sess_name],
                                               sparse_fg_list, sess_feture_list, sess_feture_list)
        # [<KerasTensor: shape=(None, 10, 4) (created by layer 'sparse_emb_13-cate_id')>, < ...'s_e_brand' >]

        ### [sparse_embedding_dict[feat](user_behavior_input_dict[sess_name][feat]) for feat in
        ###             sess_feture_list]
        keys_emb = concat_fun(keys_emb_list)  # shape=(None, 10, 8)
        tr_input.append(keys_emb)
    if bias_encoding:
        tr_input = BiasEncoding(sess_max_count)(tr_input)
    return tr_input  ## list ?????? 5 ??? (None,10,8) ??? KerasTensor

def sess_interest_extractor(tr_input, sess_max_count, TR):
    tr_out = []
    for i in range(sess_max_count):
        tr_out.append(TR(
            [tr_input[i], tr_input[i]]))
    sess_fea = concat_fun(tr_out, axis=1)
    return sess_fea
