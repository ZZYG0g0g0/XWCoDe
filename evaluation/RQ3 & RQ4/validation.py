"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2024/3/31
Last Updated: 2024/3/31
Version: 1.0.0
"""
import numpy as np
import scipy.stats as stats

if __name__ == '__main__':

    # 每个方法的F1得分
    xwc_eTour = np.array([0.5373, 0.5314, 0.5141, 0.5072, 0.4999, 0.4819, 0.4625, 0.4354, 0.4007])
    df4rt_eTour = np.array([0.6667, 0.5366, 0.4789, 0.4248, 0.4211, 0.409, 0.389, 0.3846, 0.36])


    # 弗莱德曼检验
    # 用于比较XWCoDe与每个其他方法的F1得分的显著性
    methods = [df4rt_eTour]
    method_names = ['DF4RT']
    results = {}

    for method, name in zip(methods, method_names):
        stat, p = stats.wilcoxon(xwc_eTour, method)
        results[name] = (stat, p)

    print(results)

