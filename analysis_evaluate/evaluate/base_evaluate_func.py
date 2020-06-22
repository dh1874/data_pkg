#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-29
# @Author  : HD
import numpy as np


def entropy(df):
    """

    :param df:
    :return:
    """


def gini_coefficient(self):
    """
    按某指标值升序后，进行占比累加

    :return: 基尼系数
    """

    # cum_wealths = np.cumsum(sorted(np.append(wealths_list, 0)))
    #
    # sum_wealths = cum_wealths[-1]
    #
    # xarray = np.array(range(0, len(cum_wealths))) / np.float(len(cum_wealths) - 1)
    #
    # yarray = cum_wealths / sum_wealths

    y_array = self.accum_data_frame['tpr'].tolist()

    x_array = np.array(range(0, len(y_array))) / np.float(len(y_array) - 1)

    B = np.trapz(y_array, x=x_array)

    A = 0.5 - B

    return A / (A + B)


def calc_best_n_for_k_mean(data, max_ncluster_nums):
    """
    评价kmeans聚类中心个数的最优选取

    :param max_ncluster_nums: 聚类个数的上限
    :return:
    """
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist

    distortions = []  # 各类畸变程度之和

    K = range(1, max_ncluster_nums)

    for k_ in range(1, max_ncluster_nums):  # 计算【每个k值，对应一个畸变程度】

        kmeans = KMeans(n_clusters=k_)

        kmeans.fit(data)

        distortions.append(sum(np.min(cdist(data, kmeans.cluster_centers_,
                                            'euclidean'), axis=1)) / data.shape[0])

    plt.figure()

    plt.plot(np.array(K), aa, 'bx-')

    plt.show()
