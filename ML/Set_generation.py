#!/usr/bin/python
# -*-coding:ISO8859-1-*-
# @Time : 2022/10/30 14:00
# @Author : ����
import re
import numpy as np


# ���ɲ�ѯ���򱻲�ѯ��(�������ݼ�)
def set_generation(query_file):
    # ���ɲ�ѯ���򱻲�ѯ��(�������ݼ�)
    #:param query_file: �ִʺ�ȥ��ͣ�ôʺ�����ݼ�
    #:return: ����һ���б��б��ÿ��Ԫ��Ҳ��һ���б������е��б��ÿ��Ԫ�ض���ÿһ�������еĵ��ʡ�
    with open(query_file, "r", encoding="ISO8859-1") as ft:
        lines_T = ft.readlines()
    setline = []
    for line in lines_T:
        word = line.split(' ')
        word = [re.sub('\s', '', i) for i in word]
        word = [i for i in word if len(i) > 0]
        setline.append(word)
    return setline


def tf_D(index, corpus):
    # ����tf��t,D��=����t������D�г��ֵĴ���/����D���ܴ���
    #:param : index����0��1����0������ı�ǩ
    #:param corpus: doc2bow֮��Ľ��
    sum = 0  # sum:�ô��������ĵ��г��ֵĴ���
    all = 0  # all���ĵ����ܴ���
    null = []
    for i in corpus:
        if i == null:
            all = all
        else:
            all = np.sum(np.asarray(i), 0)[1] + all
        for j in i:
            if index == j[0]:
                sum = sum + j[1]

    tf = sum / all
    # print("tf_D",tf)
    return tf


def tf_d(index, corpus):
    # ����tf��t,d��=����t���ĵ�d�г��ֵĴ���/�ĵ�d���ܴ���
    #:param : index����0��1����0������ı�ǩ
    #:param corpus: ÿһ���ĵ���doc2bow֮��Ľ��[(0,1),(1,2)]
    all = np.sum(np.asarray(corpus), 0)[1]  # all���ĵ����ܴ���
    tf = index[1] / all
    return tf


def pd(index, corpus):
    # ����p(d)=����t���ĵ�t�г��ֵĴ���/����t���ĵ�D�г��ֵĴ���
    #:param : index����0��1����0������ı�ǩ
    #:param corpus: ÿһ���ĵ���doc2bow֮��Ľ��[(0,1),(1,2)]
    sum = 0  # sum:�ô��������ĵ��г��ֵĴ���
    for i in corpus:
        for j in i:
            if index[0] == j[0]:
                sum = sum + j[1]
    pd_num = index[1] / sum
    # print(index[0],sum)
    return pd_num


def pd_entropy(corpus):
    # ����ÿ���ʵ�entropy
    # ����p(d)=����t���ĵ�t�г��ֵĴ���/����t���ĵ�D�г��ֵĴ���
    #:param : index����0��1����0������ı�ǩ
    # print(corpus)
    sum = corpus / np.sum(corpus, axis=0)  # ����p(d)

    result_doc = np.fabs(sum * (np.log(sum) / np.log(len(corpus))))  # ����ÿһ�У�����ÿ���ʵ�entropy
    result_doc[np.isnan(result_doc)] = 0
    result = np.sum(result_doc, axis=0)
    return result
