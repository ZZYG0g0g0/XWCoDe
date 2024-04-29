"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2023/10/30
Last Updated: 2023/10/30
Version: 1.0.0
"""
import os
import re
from gensim import corpora, models, similarities
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

# 生成查询集或被查询集(生成数据集)
def set_generation(query_file):
    """
    生成查询集或被查询集(生成数据集)
    :param query_file: 分词和去除停用词后的数据集
    :return: 返回一个列表，列表的每个元素也是一个列表，后者中的列表的每个元素都是每一条数据中的单词。
    """
    with open(query_file, "r", encoding="ISO-8859-1") as ft:
        lines_T = ft.readlines()
    setline = []
    for line in lines_T:
        word = line.split(' ')
        word = [re.sub('\s', '', i) for i in word]
        word = [i for i in word if len(i) > 0]
        setline.append(word)
    return setline

# VSM相似度计算
def vsm_similarity(queried_file, query_file, output_fname=None):
    # 生成被查询集
    queried_line = set_generation(queried_file)
    # 生成查询集
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]

    # 计算tfidf值
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    # 待检索的文档向量初始化一个相似度计算的对象
    corpus_sim = similarities.MatrixSimilarity(corpus_tfidf)

    # 查询集生成corpus和tfidf值
    query_corpus = [dictionary.doc2bow(text) for text in query_line]  # 在每句话中每个词语出现的频率
    query_tfidf = tfidf_model[query_corpus]
    sim = pd.DataFrame(corpus_sim[query_tfidf])
    if output_fname is not None:
        sim.to_excel(output_fname)
    return sim

# LSI相似度计算
def lsi_similarity(queried_file, query_file, output_fname=None):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]

    # 计算tfidf值
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    # 生成lsi主题
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary)
    corpus_lsi = lsi_model[corpus_tfidf]

    # 待检索的文档向量初始化一个相似度计算的对象
    corpus_sim = similarities.MatrixSimilarity(corpus_lsi)

    # 查询集生成corpus和tfidf值
    query_corpus = [dictionary.doc2bow(text) for text in query_line]  # 在每句话中每个词语出现的频率
    query_tfidf = tfidf_model[query_corpus]
    query_lsi = lsi_model[query_tfidf]
    sim = pd.DataFrame(corpus_sim[query_lsi])

    if output_fname is not None:
        sim.to_excel(output_fname)
    return sim

def HellingerDistance(p, q):
    """
    计算HellingerDistance距离
    :param p:
    :param q:
    :return: float距离
    """
    return 1 - (1 / np.sqrt(2) * np.linalg.norm(np.sqrt(p) - np.sqrt(q)))

def hellingerSim(A_set, B_set, topic_number):
    """
    计算两个集合中每条数据之间的Hellinger距离
    :param A_set: 被查询集
    :param B_set: 查询集
    :return: 一个 len(B_set) * len(A_set) 的 pandas.DataFrame
    """
    df = pd.DataFrame(index=range(len(B_set)), columns=range(len(A_set)))
    A_matrix = np.zeros((len(A_set), topic_number))
    B_matrix = np.zeros((len(B_set), topic_number))

    # 将A_set和B_set分别转化为List[List[float]](e.i. 二维矩阵)
    row = 0
    for tu in A_set:
        for i in tu:
            A_matrix[row][i[0]] = i[1]
        row = row + 1
    row = 0
    for tu in B_set:
        for i in tu:
            B_matrix[row][i[0]] = i[1]
        row = row + 1

    # 开始计算Hellinger距离
    for row in range(len(B_set)):
        for column in range(len(A_set)):
            df.iloc[[row], [column]] = HellingerDistance(B_matrix[row], A_matrix[column])  # B_matrix为查询集，所以放前面
    return df

# LDA相似度计算
def lda_similarity(queried_file, query_file, output_fname=None):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]

    # 生成lda主题
    topic_number = 100
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=topic_number, random_state=0)
    document_topic = lda_model.get_document_topics(corpus)

    # 查询集生成corpus和tfidf值
    query_corpus = [dictionary.doc2bow(text) for text in query_line]  # 在每句话中每个词语出现的频率
    query_lda = lda_model.get_document_topics(query_corpus)

    sim = hellingerSim(document_topic, query_lda, topic_number)

    if output_fname is not None:
        sim.to_excel(output_fname)
    return sim

# JS散度相似度计算
def JS_similarity(queried_file, query_file, output_fname=None):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
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

    sum_matrix = np.sum(np.vstack((A_matrix, B_matrix)), axis=0)
    probability_A = A_matrix / sum_matrix
    probability_B = B_matrix / sum_matrix

    sim = JS_Sim(probability_A, probability_B)

    if output_fname is not None:
        sim.to_excel(output_fname)
    return sim

def JS_Sim(A_set, B_set) -> pd.DataFrame:
    df = pd.DataFrame(index=range(len(B_set)), columns=range(len(A_set)))
    # 开始计算JS相似度
    for row in range(len(B_set)):
        for column in range(len(A_set)):
            df.iloc[[row], [column]] = JS_divergence(B_set[row], A_set[column])  # B_set为查询集，所以放前面
    return df

def JS_divergence(p, q):
    M = (p + q) / 2
    pk = np.asarray(p)
    pk2 = np.asarray(q)
    a = 0
    b = 0
    if (np.sum(pk, axis=0, keepdims=True) != 0):
        a = 0.5 * scipy.stats.entropy(p, M)
    if (np.sum(pk2, axis=0, keepdims=True) != 0):
        b = 0.5 * scipy.stats.entropy(q, M)
    return a + b  # 选用自然对数

def plot_pr_curve(ax, precisions, recalls, method_name, color):
    ax.plot(recalls, precisions, '-o', label=method_name, markersize=4, color=color)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid(True)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper right")

def ir_model_res(dataset_list):
    baseline_methods = [vsm_similarity, lsi_similarity, lda_similarity, JS_similarity]
    colors = ['r', 'g', 'b', 'm']  # Assign a color to each method

    # Create a large figure containing a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, dataset in enumerate(dataset_list):
        if i < 4:  # Only for the first four datasets
            ax = axes[i]  # Get the current subplot
            uc_names = os.listdir('../../datasets/' + dataset + '/req')
            cc_names = os.listdir('../../datasets/' + dataset + '/code')
            with open('../../datasets/'+dataset+'/'+dataset.lower()+ '_solution_links.txt', 'r', encoding='ISO-8859-1') as tf:
                lines = tf.readlines()
                true_set = [line.strip() for line in lines]

            for j, (baseline_method, color) in enumerate(zip(baseline_methods, colors)):
                sim = baseline_method('../../datasets/'+dataset+'/cc_doc.txt', '../../datasets/'+dataset+'/uc_doc.txt')
                result = [(row, col, sim.at[row, col]) for row in sim.index for col in sim.columns]
                sorted_result = sorted(result, key=lambda x: x[2], reverse=True)
                thresholds = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                              0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
                              0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
                space = len(sorted_result)

                precisions = []
                recalls = []
                for threshold in thresholds:
                    link_num = int(threshold * space)
                    correct = 0
                    for k in range(link_num):
                        if uc_names[sorted_result[k][0]].strip() + ': ' + cc_names[sorted_result[k][1]].strip() in true_set:
                            correct += 1
                    precision = correct / link_num if link_num else 0
                    recall = correct / len(true_set)
                    precisions.append(precision)
                    recalls.append(recall)

                # Plot the PR curve
                ax.plot(recalls, precisions, label=f'{baseline_method.__name__}', color=color)
                ax.set_title(f'{dataset}')
                ax.legend()
                ax.set_ylim(0, 1)  # Set y-axis to range from 0 to 1
    plt.tight_layout()
    plt.savefig('All_PR_Curves2x2.png')  # Save the figure to disk
    plt.show()

if __name__ == '__main__':
    dataset_list = ['eANCI', 'eTour','iTrust', 'SMOS']
    ir_model_res(dataset_list)
