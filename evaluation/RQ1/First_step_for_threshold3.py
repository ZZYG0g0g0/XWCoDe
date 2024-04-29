"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2024/3/24
Last Updated: 2024/3/24
Version: 1.0.0
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('../../'))

from ML.machine_learning import xgboost_classification_test_threshold3

def write_results_to_excel_to_t3(result, file_path):
    # 解包result元组
    test_size, results = result
    # 为当前test_size创建一个DataFrame
    test_size_df = pd.DataFrame(results, columns=["threshold3", "precision", "recall", "f1"])

    # 使用ExcelWriter，设置模式为追加模式，如果文件不存在则创建
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        # 尝试加载现有的工作表，如果不存在则创建一个新的
        try:
            # 尝试读取现有的工作表
            df_existing = pd.read_excel(writer, sheet_name='RQ1_Step1')
        except Exception as e:
            # 如果读取失败，说明工作表不存在，创建一个空的DataFrame
            df_existing = pd.DataFrame()

        # 获取已存在数据的列数，用于确定新数据应该开始的列位置
        startcol = df_existing.shape[1] + 1  # 空出一列作为分隔

        # 在同一工作表中追加数据，使用startcol根据已存在数据确定起始列
        test_size_df.to_excel(writer, sheet_name='RQ1_Step1', startcol=startcol, index=False)

def best_threshold3x(datasets):
    normalized_f1_scores = {}
    best_avg_f1 = 0
    best_config = None

    for dataset in datasets:
        res_path = f'../../res/result_{dataset}.xlsx'
        df = pd.read_excel(res_path, sheet_name='RQ1_Step1')

        # 收集该数据集所有F1值
        all_f1_values = []
        for i in range(0, 45, 5):  # 每个test_size有5列数据，共9组
            all_f1_values.extend(df.iloc[:, i + 4].values)

        # 计算平均值和标准差
        dataset_avg_f1 = np.mean(all_f1_values)
        dataset_std_f1 = np.std(all_f1_values)

        # 标准化F1值
        for i in range(0, 45, 5):
            for index, row in df.iterrows():
                threshold = row[i + 1]
                f1 = (row[i + 4] - dataset_avg_f1) / dataset_std_f1  # 标准化F1值

                if threshold not in normalized_f1_scores:
                    normalized_f1_scores[threshold] = []
                normalized_f1_scores[threshold].append(f1)

    # 确定最佳配置
    for key, scores in normalized_f1_scores.items():
        avg_f1 = np.mean(scores)  # 使用NumPy的mean函数计算平均值
        if avg_f1 >= best_avg_f1:
            best_avg_f1 = avg_f1
            best_config = key

    return best_config

def draw_threshold3(datasets):
    normalized_f1_scores = {}

    for dataset in datasets:
        res_path = f'../../res/result_{dataset}.xlsx'
        df = pd.read_excel(res_path, sheet_name='RQ1_Step1')

        # 收集该数据集所有F1值
        all_f1_values = []
        for i in range(0, 45, 5):  # 每个test_size有5列数据，共9组
            all_f1_values.extend(df.iloc[:, i + 4].values)

        # 计算平均值和标准差
        dataset_avg_f1 = np.mean(all_f1_values)
        dataset_std_f1 = np.std(all_f1_values)

        # 标准化F1值
        for i in range(0, 45, 5):
            for index, row in df.iterrows():
                threshold = row[i + 1]
                f1 = (row[i + 4] - dataset_avg_f1) / dataset_std_f1  # 标准化F1值

                if threshold not in normalized_f1_scores:
                    normalized_f1_scores[threshold] = []
                normalized_f1_scores[threshold].append(f1)

    X = []
    y = []

    for key, value in normalized_f1_scores.items():
        X.append(key)
        y.append(np.mean(value))

    # 找出最高F1分数及其对应的σ_3值
    max_f1_index = y.index(max(y))
    max_f1 = y[max_f1_index]
    max_f1_sigma_3 = X[max_f1_index]

    #画折线图
    plt.figure(figsize=(10, 6))
    plt.plot(X, y, color='b')

    # 获取当前Y轴的范围
    ymin, _ = plt.ylim()

    # 从最高点画一条虚线至X轴
    plt.vlines(x=max_f1_sigma_3,ymin=ymin, ymax=max_f1, color='r', linestyle='--')

    text_x_position = max_f1_sigma_3 + 0.05  # 根据需要调整这个偏移量

    plt.text(text_x_position, max_f1, r'$\sigma_3={}$'.format(max_f1_sigma_3), horizontalalignment='left',
                 verticalalignment='center', color='red')

    plt.xlabel(r'$\sigma_3$')
    plt.ylabel('normalized F1 Score')
    # 显示网格
    plt.grid(True)

    # 显示图例
    plt.legend(['Average F1 Score'])

    # 保存为PDF格式
    plt.savefig('average_f1_score_by_sigma_3.pdf', format='pdf')
    # 展示图表
    plt.show()


if __name__ == '__main__':
    datasets = ['iTrust', 'smos', 'eTour', 'eANCI']
    #step1
    # test_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # for dataset in datasets:
    #     print(f"RQ1_Step1->dataset:{dataset}")
    #     res = xgboost_classification_test_threshold3(test_size, dataset, "selected_features")
    #     for r in res:
    #         if dataset == 'eANCI':
    #             write_results_to_excel_to_t3(r, f'../../res/result_eANCI.xlsx')
    #         if dataset == 'eTour':
    #             write_results_to_excel_to_t3(r, f'../../res/result_eTour.xlsx')
    #         if dataset == 'iTrust':
    #             write_results_to_excel_to_t3(r, f'../../res/result_iTrust.xlsx')
    #         if dataset == 'smos':
    #             write_results_to_excel_to_t3(r, f'../../res/result_smos.xlsx')

    #step2
    # best_config = best_threshold3x(datasets)
    # print(best_config)

    #step3
    draw_threshold3(datasets)