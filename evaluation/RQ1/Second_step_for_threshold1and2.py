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

from ML.machine_learning import xgboost_classification_test1and2
from ML.machine_learning import calculate_dependency_importance

def write_results_to_excel(result, file_path):
    # 解包result元组
    test_size, results = result
    # 为当前test_size创建一个DataFrame
    test_size_df = pd.DataFrame(results, columns=["threshold1", "threshold2", "precision", "recall", "f1"])

    # 使用ExcelWriter，设置模式为追加模式，如果文件不存在则创建
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        # 尝试加载现有的工作表，如果不存在则创建一个新的
        try:
            # 尝试读取现有的工作表
            df_existing = pd.read_excel(writer, sheet_name='RQ1_Step2')
        except Exception as e:
            # 如果读取失败，说明工作表不存在，创建一个空的DataFrame
            df_existing = pd.DataFrame()

        # 获取已存在数据的列数，用于确定新数据应该开始的列位置
        startcol = df_existing.shape[1] + 1  # 空出一列作为分隔

        # 在同一工作表中追加数据，使用startcol根据已存在数据确定起始列
        test_size_df.to_excel(writer, sheet_name='RQ1_Step2', startcol=startcol, index=False)

def best_threshold1and2x(datasets):
    normalized_f1_scores = {}
    best_avg_f1 = 0
    best_config = None

    for dataset in datasets:
        res_path = f'../../res/result_{dataset}.xlsx'
        df = pd.read_excel(res_path, sheet_name='RQ1_Step2')
        # 收集该数据集所有F1值
        all_f1_values = []
        for i in range(0, 54, 6):  # 每个test_size有5列数据，共9组
            all_f1_values.extend(df.iloc[:, i + 5].values)
        # 计算平均值和标准差
        dataset_avg_f1 = np.mean(all_f1_values)
        dataset_std_f1 = np.std(all_f1_values)

        # 标准化F1值
        for i in range(0, 54, 6):
            for index, row in df.iterrows():
                threshold1 = row[i + 1]
                threshold2 = row[i + 2]
                f1 = (row[i + 5] - dataset_avg_f1) / dataset_std_f1  # 标准化F1值

                key = (threshold1, threshold2)
                if key not in normalized_f1_scores:
                    normalized_f1_scores[key] = []
                normalized_f1_scores[key].append(f1)

        # 确定最佳配置
    for key, scores in normalized_f1_scores.items():
        avg_f1 = np.mean(scores)  # 使用NumPy的mean函数计算平均值
        if avg_f1 >= best_avg_f1:
            best_avg_f1 = avg_f1
            best_config = key

    return best_config

def draw_threshold1and2(datasets):
    normalized_f1_scores = {}

    for dataset in datasets:
        res_path = f'../../res/result_{dataset}.xlsx'
        df = pd.read_excel(res_path, sheet_name='RQ1_Step2')
        # 收集该数据集所有F1值
        all_f1_values = []
        for i in range(0, 54, 6):  # 每个test_size有5列数据，共9组
            all_f1_values.extend(df.iloc[:, i + 5].values)
        # 计算平均值和标准差
        dataset_avg_f1 = np.mean(all_f1_values)
        dataset_std_f1 = np.std(all_f1_values)

        # 标准化F1值
        for i in range(0, 54, 6):
            for index, row in df.iterrows():
                threshold1 = row[i + 1]
                threshold2 = row[i + 2]
                f1 = (row[i + 5] - dataset_avg_f1) / dataset_std_f1  # 标准化F1值

                key = (threshold1, threshold2)
                if key not in normalized_f1_scores:
                    normalized_f1_scores[key] = []
                normalized_f1_scores[key].append(f1)

    x = []
    y = []
    z = []
    # 确定最佳配置
    for key, scores in normalized_f1_scores.items():
        avg_f1 = np.mean(scores)  # 使用NumPy的mean函数计算平均值
        x.append(key[0])
        y.append(key[1])
        z.append(avg_f1)

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维曲面图
    surf = ax.plot_trisurf(x, y, z, cmap='viridis')

    # 找出Z值最大的点
    max_z_index = np.argmax(z)
    max_x = x[max_z_index]
    max_y = y[max_z_index]
    max_z = z[max_z_index]
    # 在最大Z值的点绘制一个红色的点
    ax.scatter(max_x, max_y, max_z, color='red', s=10)  # s是点的大小
    # 在图中标注最大Z值点的(X, Y)坐标
    ax.text(max_x, max_y, max_z, f'({max_x:.2f}, {max_y:.2f})', color='red')

    # 添加坐标轴标签
    ax.set_xlabel(r'$\sigma_1$')
    ax.set_ylabel(r'$\sigma_2$')
    ax.set_zlabel('Normalized Average F1 Score')

    # 添加颜色条
    fig.colorbar(surf)

    # 旋转视角以避免遮挡
    ax.view_init(elev=20., azim=45)

    # 显示图形
    plt.show()

    #保存为PDF
    fig.savefig('./average_f1_score_by_sigma_1and2.pdf', format='pdf')


if __name__ == '__main__':
    #训练结果
    datasets = ['smos', 'eTour', 'eANCI','iTrust']
    # test_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # for dataset in datasets:
    #     ci, mi, wi, dependency_matrix = calculate_dependency_importance(dataset, datasets)
    #     res = xgboost_classification_test1and2(test_size, dataset, "selected_features", ci, mi, wi, dependency_matrix)
    #     for r in res:
    #         if dataset == 'eANCI':
    #             write_results_to_excel(r, '../../res/result_eANCI.xlsx')
    #         if dataset == 'eTour':
    #             write_results_to_excel(r, '../../res/result_eTour.xlsx')
    #         if dataset == 'iTrust':
    #             write_results_to_excel(r, '../../res/result_iTrust.xlsx')
    #         if dataset == 'smos':
    #             write_results_to_excel(r, '../../res/result_smos.xlsx')

    #2.找到最佳参数
    # best_config = best_threshold1and2x(datasets)
    # print(best_config)

    #3.绘制图形
    draw_threshold1and2(datasets)