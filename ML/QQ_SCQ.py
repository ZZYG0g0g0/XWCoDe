# -*-coding:utf-8-*-
# @Time : 2022/10/10 20:44
# @Author : �������Ҹ�
import pandas as pd

import warnings
import SCQ

warnings.filterwarnings(action='ignore')


def QQ_feature_generation(fname, tname, output_fname=None):
    QQ_feature = pd.DataFrame()
    code_feature = ['CN_MN_VN_CMT']
    flag = 0
    options = [SCQ.calcSCQList_final]
    # print(options)
    for option in options:
        df_avg, df_sum, df_max = option(fname + code_feature[0] + '_clear.txt', tname)  # tnameΪ��ѯ����fnameΪ����ѯ��
        QQ_feature[flag] = pd.concat([df_avg.iloc[i] for i in range(df_avg.shape[0])], axis=0,
                                     ignore_index=True)  # ��sim������ת��ΪIR_based_feature��һ��
        # print(option)
        flag = flag + 1
        QQ_feature[flag] = pd.concat([df_sum.iloc[i] for i in range(df_sum.shape[0])], axis=0,
                                     ignore_index=True)  # ��sim������ת��ΪIR_based_feature��һ��
        flag = flag + 1
        QQ_feature[flag] = pd.concat([df_max.iloc[i] for i in range(df_max.shape[0])], axis=0,
                                     ignore_index=True)  # ��sim������ת��ΪIR_based_feature��һ��
        flag = flag + 1

        df_avg, df_sum, df_max = option(tname, fname + code_feature[0] + '_clear.txt')  # ��ѯ���ͱ���ѯ������
        QQ_feature[flag] = pd.concat([df_avg.iloc[:, i] for i in range(df_avg.shape[1])], axis=0,
                                     ignore_index=True)  # ��sim������ת��ΪIR_based_feature��һ��
        flag = flag + 1

        QQ_feature[flag] = pd.concat([df_sum.iloc[:, i] for i in range(df_sum.shape[1])], axis=0,
                                     ignore_index=True)  # ��sim������ת��ΪIR_based_feature��һ��
        flag = flag + 1

        QQ_feature[flag] = pd.concat([df_max.iloc[:, i] for i in range(df_max.shape[1])], axis=0,
                                     ignore_index=True)  # ��sim������ת��ΪIR_based_feature��һ��
        flag = flag + 1

    if output_fname is not None:
        QQ_feature.to_excel(output_fname + ".xlsx")
    return QQ_feature


if __name__ == '__main__':
    fname = "../easyclinic_ID_UC/"
    tname = "../easyclinic_ID_UC/uc_clear.txt"
    output_fname = "../easyclinic_ID_UC/SCQ"
    QQ_feature_generation(fname, tname, output_fname)
