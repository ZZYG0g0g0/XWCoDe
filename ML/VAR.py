# -*-coding:GBK-*-
# @Time : 2022/11/14 22:00 over
# @Author : ����
import numpy as np
import pandas as pd
from gensim import corpora
from gensim import models
import sys

sys.path.append('Set_generation.py')
sys.path.append('Specificity_IDF.py')
import Set_generation
import Specificity_IDF
import numpy as np


# ����w(t,d)
def w(queried_file, query_file):
    # ����idf(t)
    IDF_result = Specificity_IDF.IDF(queried_file, query_file)
    # print("IDF_result",IDF_result)
    query_line = Set_generation.set_generation(query_file)
    queried_line = Set_generation.set_generation(queried_file)
    # ���ɴʵ�ͼ����Ƶcorpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in query_line]
    # ����tf(t,d)
    tf_resutl = []
    for i in corpus:
        tf_sentence = []
        for j in i:
            tf = Set_generation.tf_d(j, i)
            tf_sentence.append((j[0], tf))

        tf_resutl.append(tf_sentence)

    # print("tf_resutl",tf_resutl)
    # ����w��t,d��
    w_result = []
    for i in range(len(tf_resutl)):
        w_sentence = []
        for j in range(len(tf_resutl[i])):
            w = tf_resutl[i][j][1] * IDF_result[i][j][1]
            w_sentence.append((tf_resutl[i][j][0], w))
            # print(IDF_result[i][j][1])
        w_result.append(w_sentence)
    #print("w_result:", w_result)
    return w_result


# ȥ�������ĵ���ȫ�����ֵĴʣ�Ȼ�����tfidf
# tfidf���ƶȼ���
def tf_idf(queried_file, query_file, output_fname=None):
    queried_line = Set_generation.set_generation(queried_file)
    query_line = Set_generation.set_generation(query_file)
    # ����ѯ�����ɴʵ��corpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in query_line]
    # ����tfidfֵ
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]
    return corpus_tfidf,len(queried_line),len(query_line)


# ����VAR(��׼��)
def VAR(queried_file, query_file):
    # ����tfidf
    w, colum,row  = tf_idf(queried_file, query_file)
    # �ҵ�������
    max = w[0][0][0]
    small = w[0][0][0]

    w_num = 0
    for i in w:
        w_num = w_num + 1
        for j in i:
            if max < j[0]:
                max = j[0]
            if small > j[0]:
                small = j[0]
    # print(np.shape(w)[0])
    if small == 0:
        max = max + 1
    matrix = np.zeros((max, w_num))  # 16*4
    # print((max,w_num))
    # ��tfidf���ת��Ϊ����
    for i in range(len(w)):  # ��
        for j in range(len(w[i])):  # ��
            hang = w[i][j][0] - small
            lie = i
            # print(hang,lie)
            matrix[hang][lie] = w[i][j][1]
            print(hang, lie)

    # print("var:",matrix)
    # ����ÿһ�в�Ϊ0�ķ���
    dictionary = var2(matrix)
    matrix = matrix.transpose()
    result = []
    for i in range(len(matrix)):
        sentence = []
        for j in range(len(matrix[i])):
            if matrix[i][j] > 0:
                sentence.append((j, dictionary[j]))
        result.append(sentence)
    print("var:", result)

    return result, colum,row


def var2(maxtrix):
    print(maxtrix)
    dictionary = dict()
    for i in range(len(maxtrix)):
        sum = 0
        all = 0
        avg_all = np.sum(maxtrix[i])
        for j in range(len(maxtrix[i])):
            if maxtrix[i][j] > 0:
                all = all + 1
        for j in range(len(maxtrix[i])):
            if maxtrix[i][j] > 0:
                sum = sum + (maxtrix[i][j] - avg_all / all) ** 2
            # �洢���
        if all != 0:
            dictionary.update({i: np.sqrt(sum / all)})
        else:
            dictionary.update({i: 0})
        # print(dictionary)

    return dictionary


# ����VAR��ƽ��ֵ
def AvgAVR(queried_file, query_file):
    result, colum, row = VAR(queried_file, query_file)
    avg_result = []
    for i in result:
        if not np.isnan(i).all():
            var = np.asarray(i)
            # print(idf)
            avg_var = np.average(var, 0)   # ������ȡ���ֵ (2, 0.17609125905568124)
            avg_result.append(avg_var[1])
        else:
            avg_result.append(0)

    df_avg = pd.DataFrame(index=range(row), columns=range(colum))
    for i in range(row):
        for j in range(colum):
            df_avg.iloc[i][j] = avg_result[i]
    return df_avg


# ����VAR�����ֵ
def MaxVAR(queried_file, query_file):
    result, colum,row  = VAR(queried_file, query_file)
    max_result = []
    for i in result:
        if not np.isnan(i).all():
            var = np.asarray(i)
            max_var = np.amax(var, 0)  # ������ȡ���ֵ (2, 0.17609125905568124)
            max_result.append(max_var[1])
        else:
            max_result.append(0)

    df_avg = pd.DataFrame(index=range(row), columns=range(colum))
    for i in range(row):
        for j in range(colum):
            df_avg.iloc[i][j] = max_result[i]
    return df_avg



# ����VAR���ܺ�
def SumVAR(queried_file, query_file):
    result, colum, row = VAR(queried_file, query_file)
    sum_result = []
    for i in result:
        if not np.isnan(i).all():
            var = np.asarray(i)
            sum_var = np.average(var, 0)  # ������ȡ���ֵ (2, 0.17609125905568124)
            sum_result.append(sum_var[1])
        else:
            sum_result.append(0)

    df_sum = pd.DataFrame(index=range(row), columns=range(colum))
    for i in range(row):
        for j in range(colum):
            df_sum.iloc[i][j] = sum_result[i]
    return df_sum


if __name__ == "__main__":
    query_file = "./UC_clear.txt"
    queried_file = "./CN_MN_VN_CMT_clear.txt"
    output_fname = "./QQ"
    MaxVAR(queried_file, query_file)
# AvgAVR(queried_file,query_file)
# MaxVAR(queried_file,query_file)
# SumVAR(queried_file,query_file)
# df_idf(queried_file, query_file)
