#!/usr/bin/python
# -*-coding:utf-8-*-
# @Time : 2022/10/28 20:33
# @Author : ���� over
from gensim import corpora
import numpy as np
import sys
import pandas as pd

sys.path.append('Set_generation.py')
import Set_generation
import warnings

warnings.filterwarnings(action='ignore')


def count(index, corpus):
    # ����ô����ĵ��г��ֵĴ���
    num = 0
    for i in corpus:  # [(1,2),(2,3)]
        for j in i:  # (1,2)
            if str(index) == str(j[0]):
                num = num + 1
                break
    return num


def IDF(queried_file, query_file):
    # ����IDF

    queried_line = Set_generation.set_generation(queried_file)
    query_line = Set_generation.set_generation(query_file)
    # ���ɴʵ�ͼ����Ƶcorpus

    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in query_line]

    all = len(query_line)
    result = []

    for i in corpus:
        sentence = []
        for j in i:
            num = count(j[0], corpus)  # num�������ôʵ��ĵ���
            idf = np.abs(np.log(all / (num + 1)))  # all���ĵ�����
            sentence.append((j[0], idf))
        result.append(sentence)
    print("IDF", result)
    row = len(query_line)
    colum = len(queried_line)
    return result, row, colum


def AvgIDF(queried_file, query_file):
    # ����ÿ���ı���IDFƽ��ֵ

    result, row, colum = IDF(queried_file, query_file)
    df = pd.DataFrame(index=range(row), columns=range(colum))
    null = []
    for i in range(row):  # ÿһ����ѯ��
        if result[i] == null:
            avg_idf = (0, 0)
        else:
            idf = np.asarray(result[i])
            avg_idf = np.mean(idf, 0)

        for j in range(colum):
            df.iloc[i][j] = avg_idf[1]
    # print("Avgidf", df)
    return df


def MaxIDF(queried_file, query_file):
    # ����ÿ���ı���IDF���ֵ

    result, row, colum = IDF(queried_file, query_file)
    df = pd.DataFrame(index=range(row), columns=range(colum))
    null = []
    for i in range(row):  # ÿһ����ѯ��
        if result[i] == null:
            amax_idf = (0, 0)
        else:
            idf = np.asarray(result[i])
            amax_idf = np.amax(idf, 0)

        for j in range(colum):
            df.iloc[i][j] = amax_idf[1]
    # print("amax", df)
    return df


def DevIDF(queried_file, query_file):
    # ����ÿ���ı���IDF��׼��
    result, row, colum = IDF(queried_file, query_file)
    df = pd.DataFrame(index=range(row), columns=range(colum))
    null = []
    for i in range(row):  # ÿһ����ѯ��
        if result[i] == null:
            std_idf = (0, 0)
        else:
            idf = np.asarray(result[i])
            std_idf = np.std(idf, 0)
        for j in range(colum):
            df.iloc[i][j] = std_idf[1]
    print("std", df)
    return df


if __name__ == "__main__":
    query_file = "./UC_clear.txt"
    queried_file = "./CN_MN_VN_CMT_clear.txt"
    output_fname = "iTrust/"
    IDF(queried_file, query_file)
    # MaxIDF(queried_file,query_file)
    # AvgIDF(queried_file,query_file)
