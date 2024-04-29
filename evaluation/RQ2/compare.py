"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2024/3/25
Last Updated: 2024/3/25
Version: 1.0.0
"""
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath('../../'))

from ML.machine_learning import xgboost_classification_kfold, XGB_CD_GE, calculate_dependency_importance, XGB_CD, other_classification_model

if __name__ == '__main__':
    # 初始化一个空的DataFrame用于存储结果
    results_list = []

    datasets = ['eANCI', 'eTour', 'iTrust', 'smos']
    for dataset in datasets:
        res_XGB = xgboost_classification_kfold(dataset, "selected_features")
        print(f'res_XGB:{dataset}->{res_XGB}')

        ci, mi, wi, dependency_matrix = calculate_dependency_importance(dataset, datasets)
        res_XGB_CD_GE = XGB_CD_GE(dataset, "selected_features", ci, mi, wi, dependency_matrix)
        print(f'res_XGB_CD_GE:{dataset}->{res_XGB_CD_GE}')
        res_XGB_CD = XGB_CD(dataset, "selected_features", dependency_matrix)
        print(f'res_XGB_CD:{dataset}->{res_XGB_CD}')
        # 将结果添加到列表
        results_list.append({'Dataset': dataset,
                             'XGB': round(res_XGB, 4),
                             'XGB_CD': round(res_XGB_CD, 4),
                             'XGB_CD_GE': round(res_XGB_CD_GE, 4)})

    # 将列表转换为DataFrame
    results_df = pd.DataFrame(results_list)
    # 将DataFrame写入Excel文件
    results_df.to_excel('./albation_results.xlsx', index=False)
