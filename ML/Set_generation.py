#!/usr/bin/python
# -*-coding:ISO8859-1-*-
# @Time : 2022/10/30 14:00
# @Author : 邓洋
import re
import numpy as np


# 生成查询集或被查询集(生成数据集)
def set_generation(query_file):
    # 生成查询集或被查询集(生成数据集)
    #:param query_file: 分词和去除停用词后的数据集
    #:return: 返回一个列表，列表的每个元素也是一个列表，后者中的列表的每个元素都是每一条数据中的单词。
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
    # 计算tf（t,D）=术语t在文章D中出现的次数/文章D的总词数
    #:param : index：（0，1）的0，词语的标签
    #:param corpus: doc2bow之后的结果
    sum = 0  # sum:该词在所有文档中出现的次数
    all = 0  # all：文档的总词数
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
    # 计算tf（t,d）=术语t在文档d中出现的次数/文档d的总词数
    #:param : index：（0，1）的0，词语的标签
    #:param corpus: 每一个文档的doc2bow之后的结果[(0,1),(1,2)]
    all = np.sum(np.asarray(corpus), 0)[1]  # all：文档的总词数
    tf = index[1] / all
    return tf


def pd(index, corpus):
    # 计算p(d)=术语t在文档t中出现的次数/术语t在文档D中出现的次数
    #:param : index：（0，1）的0，词语的标签
    #:param corpus: 每一个文档的doc2bow之后的结果[(0,1),(1,2)]
    sum = 0  # sum:该词在所有文档中出现的次数
    for i in corpus:
        for j in i:
            if index[0] == j[0]:
                sum = sum + j[1]
    pd_num = index[1] / sum
    # print(index[0],sum)
    return pd_num


def pd_entropy(corpus):
    # 计算每个词的entropy
    # 计算p(d)=术语t在文档t中出现的次数/术语t在文档D中出现的次数
    #:param : index：（0，1）的0，词语的标签
    # print(corpus)
    sum = corpus / np.sum(corpus, axis=0)  # 计算p(d)

    result_doc = np.fabs(sum * (np.log(sum) / np.log(len(corpus))))  # 根据每一列，计算每个词的entropy
    result_doc[np.isnan(result_doc)] = 0
    result = np.sum(result_doc, axis=0)
    return result
