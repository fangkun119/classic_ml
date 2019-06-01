# !/usr/bin/python
# -*- coding:utf-8 -*-

from sklearn import metrics


if __name__ == "__main__":
    # homogeneity_score  : 1 cluster only has examples from 1 class
    # completeness_score : examples belong to 1 class only in 1 cluster
    # v_measure_score    : weighted average of homogeneity_score and completeness_score

    y = [0, 0, 0, 1, 1, 1]
    y_hat = [0, 0, 1, 1, 2, 2]
    print("y    : ", y)
    print("y_hat: ", y_hat)
    h = metrics.homogeneity_score(y, y_hat)
    c = metrics.completeness_score(y, y_hat)
    print('同一性(Homogeneity)：', h)
    print('完整性(Completeness)：', c)
    v2 = 2 * c * h / (c + h)
    v = metrics.v_measure_score(y, y_hat)
    print('V-Measure：', v2, v)

    print()
    y = [0, 0, 0, 1, 1, 1]
    y_hat = [0, 0, 1, 3, 3, 3]
    print("y    : ", y)
    print("y_hat: ", y_hat)
    h = metrics.homogeneity_score(y, y_hat)
    c = metrics.completeness_score(y, y_hat)
    v = metrics.v_measure_score(y, y_hat)
    print('同一性(Homogeneity)：', h)
    print('完整性(Completeness)：', c)
    print('V-Measure：', v)

    # 允许不同值
    print()
    y = [0, 0, 0, 1, 1, 1]
    y_hat = [1, 1, 1, 0, 0, 0]
    print("y    : ", y)
    print("y_hat: ", y_hat)    
    h = metrics.homogeneity_score(y, y_hat)
    c = metrics.completeness_score(y, y_hat)
    v = metrics.v_measure_score(y, y_hat)
    print('同一性(Homogeneity)：', h)
    print('完整性(Completeness)：', c)
    print('V-Measure：', v)

    # ARI
    print()
    y = [0, 0, 1, 1]
    y_hat = [0, 1, 0, 1]
    print("y    : ", y)
    print("y_hat: ", y_hat)    
    ari = metrics.adjusted_rand_score(y, y_hat)
    print("ari  : ", ari)

    print()
    y = [0, 0, 0, 1, 1, 1]
    y_hat = [0, 0, 1, 1, 2, 2]
    print("y    : ", y)
    print("y_hat: ", y_hat)    
    ari = metrics.adjusted_rand_score(y, y_hat)
    print("ari  : ", ari)
