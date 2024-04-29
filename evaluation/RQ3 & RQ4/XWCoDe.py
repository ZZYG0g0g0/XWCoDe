"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2024/3/26
Last Updated: 2024/3/26
Version: 1.0.0
"""
#XWCoDe
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath('../../'))

from ML.machine_learning import calculate_dependency_importance, XGB_CD_GE_by_testsize

if __name__ == '__main__':
    # 初始化一个空的DataFrame用于存储结果
    results_list = []

    datasets = ['eANCI', 'eTour', 'iTrust', 'smos']
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for dataset in datasets:
        ci, mi, wi, dependency_matrix = calculate_dependency_importance(dataset, datasets)
        for test_size in test_sizes:
            res = XGB_CD_GE_by_testsize(dataset, "selected_features", ci, mi, wi, dependency_matrix, test_size)
            # 将每次实验的结果添加到列表中
            results_list.append({'Dataset': dataset, 'Test Size': test_size, 'F1 Score': res})
    # 将列表转换为 DataFrame
    results_df = pd.DataFrame(results_list)

    # 将 DataFrame 写入 Excel 文件
    results_df.to_excel('XWCoDe_results.xlsx', index=False)