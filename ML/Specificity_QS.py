#!/usr/bin/python
# -*-coding:utf-8-*-
# @Time : 2022/10/31 14:00
# @Author : ����
from gensim import corpora
import sys

sys.path.append('Set_generation.py')
import Set_generation
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(action='ignore')


# ����ѯ��Ϊ���Ͽ�
# QS=������ѯ�ʵ��ĵ�����/���ĵ�����
def QS(queried_file, query_file):
    # ����IDF
    queried_line = Set_generation.set_generation(queried_file)
    query_line = Set_generation.set_generation(query_file)
    # ���ɲ�ѯ���ʵ�ͼ����Ƶcorpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in query_line]  # �ĵ��а������Ͽ��Ƶ��
    # print(corpus)
    QS_result = []
    D = len(query_file)
    for i in corpus:
        num = count(i, corpus)  # �����ò�ѯ�ʵ��ĵ�����
        # print(i)
        QS_result.append(num / D)
    row = len(query_line)
    colum = len(queried_line)
    df = pd.DataFrame(index=range(row), columns=range(colum))
    for i in range(row):  # ÿһ����ѯ��
        for j in range(colum):
            df.iloc[i][j] = QS_result[i]
    # print("QS_result",df)
    return df


# ��������ò�ѯ��t���ĵ�d����
def count(index, corpus):
    num = 0
    list1 = []
    list2 = []
    for i in index:
        list1.append(i[0])
        list2.append(i[1])
    index_dict = dict(zip(list1, list2))
    for i in corpus:
        for j in i:
            if (j[0] in index_dict):
                num = num + 1
                break
    return num


# SCS KLɢ��
def KL_similarity(queried_file, query_file, output_fname=None):
    queried_line = Set_generation.set_generation(queried_file)
    query_line = Set_generation.set_generation(query_file)

    # ����ѯ�����ɴʵ��corpus
    dictionary = corpora.Dictionary(queried_line + query_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]
    corpus2 = [dictionary.doc2bow(text) for text in query_line]

    A_matrix = np.zeros((len(queried_line), len(dictionary)))
    B_matrix = np.zeros((len(query_line), len(dictionary)))

    row = 0
    for document in corpus:
        for word_id, freq in document:
            A_matrix[row][word_id] = freq
        row = row + 1

    row = 0
    for document in corpus2:
        for word_id, freq in document:
            B_matrix[row][word_id] = freq
        row = row + 1
    # print(corpus)
    sum_matrix = np.sum(np.vstack((A_matrix, B_matrix)), axis=0)
    probability_A = A_matrix / sum_matrix
    probability_B = B_matrix / sum_matrix

    sim = KL_Sim(probability_A, probability_B)
    # print("KL_sim:",sim)
    if output_fname is not None:
        sim.to_excel(output_fname)
    return sim


def KL_Sim(A_set, B_set) -> pd.DataFrame:
    df = pd.DataFrame(index=range(len(B_set)), columns=range(len(A_set)))
    # ��ʼ����KL���ƶ�
    for row in range(len(B_set)):
        for column in range(len(A_set)):
            df.iloc[[row], [column]] = KL_divergence(B_set[row], A_set[column])  # B_setΪ��ѯ�������Է�ǰ��
    return df


def KL_divergence(p, q):
    pk = np.asarray(p)
    pk2 = np.asarray(q)
    kl = 0
    for i in range(len(pk)):
        if ((pk[i] != 0) & (pk2[i] != 0)):
            kl = kl + pk[i] * np.log(pk[i] / pk2[i])
            # all=all+1
    return kl


if __name__ == "__main__":
    query_file = "./UC_clear.txt"
    queried_file = "./CN_MN_VN_CMT_clear.txt"
    output_fname = "iTrust/"
    QS(queried_file, query_file)
    # KL_similarity(queried_file,query_file)
