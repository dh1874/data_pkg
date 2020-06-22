#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-27
# @Author  : HD

import numpy as np
from scipy.stats import chi2
from sklearn.cluster import KMeans
import pandas as pd

from base_calculator import BaseCalculator
from utils.example_data import ExampleData

"""
生成切分点
"""


class BaseBinningStrategy(BaseCalculator):
    """
    分箱基础策略类
    """

    # 最大分箱数
    __max_bin = 10

    def calc(self, df, feature):
        """
        :param df: df[feature]  待分箱数据处理的data_frame
        :param feature: 待分箱的特征
        :return:
        """

        pass

    @property
    def max_bin(self):
        return self.__max_bin

    @max_bin.setter
    def max_bin(self, value):
        self.__max_bin = value


class EqualWidthBinning(BaseBinningStrategy):
    """
    等寬分箱
    """

    def calc(self, df: pd.DataFrame, feature):
        """
        等距分箱
        :return:
        """

        max_value = df[feature].max()

        min_value = df[feature].min()

        interval = (max_value - min_value) / self.max_bin

        return [round(min_value, 2)] + [round(min_value + i * interval, 2) for i in range(1, self.max_bin + 1)]


class EqualFrequencyBinning(BaseBinningStrategy):
    """
    等频分箱, 默认得到最大值
    """

    # def calc(self, df: pd.DataFrame, feature):
    #
    #     # 分位数组
    #     breakpoints = [i/self.max_bin for i in range(1, self.max_bin + 1)]  # 等距百分比
    #
    #     res = df[feature].quantile(breakpoints, interpolation="lower").drop_duplicates().tolist() + [df[feature].max()]
    #
    #     return list(set(res))

    def calc(self, df:pd.DataFrame, feature):

        # 对feature, 用百分位数的形式进行排序
        df['rank'] = df[feature].rank(pct=True)

        # 等频分位数组
        breakpoints = [i / self.max_bin for i in range(1, self.max_bin + 1)]

        # 返回每个分为数组，feature的最大值，即为切分点
        return [df[df['rank'] <= i][feature].max() for i in breakpoints]


class BinningByPercentile(BaseBinningStrategy):
    """
    按指定的百分位数，返回切分点
    """

    # 取区间最大
    __get_max = False

    # 取区间最小
    __get_min = True

    percentile_point = None

    def calc(self, df, feature):

        # 对feature, 用百分位数的形式进行排序
        assert type(self.percentile_point) is list, '先配置指定的百分位数数组'

        # <= max
        if self.__get_max:

            # 默认升序， 百分位数排序
            df['rank'] = df[feature].rank(pct=True)

            self.percentile_point = sorted(self.percentile_point)

            return [df[df['rank'] <= i][feature].max() for i in self.percentile_point]

        # >= min
        elif self.__get_min:

            df['rank'] = df[feature].rank(ascending=False, pct=True)

            self.percentile_point = sorted(self.percentile_point, reverse=True)

            return [df[df['rank'] <= i][feature].min() for i in self.percentile_point]

    def set_max_switch(self):
        """
        选择分位数方式：<= max
        :return:
        """
        self.__get_max = True
        self.__get_min = False
        return self

    def set_min_switch(self):
        """
        选择分位数方式： >= min
        :return:
        """
        self.__get_max = False
        self.__get_min = True
        return self


class KMeansBinning(BaseBinningStrategy):
    """
    K均值质心，当切分点
    """
    def __init__(self, n_cluster):
        # 聚类的质心个数
        self.n_cluster = n_cluster

    def calc(self, df, feature):

        km = KMeans(n_clusters=self.n_cluster).fit(df[[feature]])

        # 质心数组
        centers = km.cluster_centers_

        min_value = round(df[feature].min(), 2)

        max_value = round(df[feature].max(), 2)

        # 升序排列的聚类中心点
        centers_list = [min_value] + sorted([round(i[0], 2) for i in centers]) + [max_value]

        return centers_list


class Chi2Binning(BaseBinningStrategy):
    """
    基于ChiMerge的卡方离散化方法

    """
    __sig_level = 0.05  # 显著性水平(significance level) = 1 - 置信度

    group_value_list = []

    def __init__(self, label_name):
        super(BaseBinningStrategy, self).__init__()
        # 标签名
        self.label_name = label_name

    @property
    def sig_level(self):
        return self.__sig_level

    @sig_level.setter
    def sig_level(self, value):
        self.__sig_level = value

    @staticmethod
    def calc_chi2(count:pd.DataFrame, group1, group2):
        """
        根据分组信息（group）计算各分组的卡方值
        :param count: DataFrame 待分箱变量各取值的正负样本数
        :param group1: list 单个分组信息
        :param group2: list 单个分组信息
        :return: 该分组的卡方值
        """

        count_intv1 = count.loc[count.index.isin(group1)].sum(axis=0).values
        count_intv2 = count.loc[count.index.isin(group2)].sum(axis=0).values
        count_intv = np.vstack((count_intv1, count_intv2))

        # 计算四联表
        row_sum = count_intv.sum(axis=1)
        col_sum = count_intv.sum(axis=0)
        total_sum = count_intv.sum()

        # 计算期望样本数
        count_exp = np.ones(count_intv.shape) * col_sum / total_sum
        count_exp = (count_exp.T * row_sum).T

        # 计算卡方值
        chi2 = (count_intv - count_exp) ** 2 / count_exp
        chi2[count_exp == 0] = 0
        return chi2.sum()

    def calc(self, df, feature):

        print("ChiMerge分箱开始：")
        count = pd.crosstab(df[feature], df[self.label_name])
        deg_freedom = len(count.columns) - 1  # 自由度(degree of freedom)= y类别数-1
        chi2_threshold = chi2.ppf(1 - self.__sig_level, deg_freedom)  # 卡方阈值
        group_value_list = np.array(count.index).reshape(-1, 1).tolist()  # 分组信息

        # 2. 计算相邻分组的卡方值
        chi2_list = [self.calc_chi2(count, group_value_list[idx], group_value_list[idx + 1])
                        for idx in range(len(group_value_list) - 1)]

        # 3. 合并相似分组并更新卡方值
        while 1:
            if min(chi2_list) >= chi2_threshold:
                print("最小卡方值%.3f大于卡方阈值%.3f，分箱合并结束！！！" % (min(chi2_list), chi2_threshold))
                break
            if len(group_value_list) <= self.max_bin:
                print("分组长度%s等于指定分组数%s" % (len(group_value_list), self.max_bin))
                break

            min_idx = chi2_list.index(min(chi2_list))
            # 根据卡方值合并卡方值最小的相邻分组
            group_value_list[min_idx] = group_value_list[min_idx] + group_value_list[min_idx + 1]
            group_value_list.remove(group_value_list[min_idx + 1])

            # 更新卡方值
            if min_idx == 0:
                chi2_list.pop(min_idx)
                chi2_list[min_idx] = self.calc_chi2(count, group_value_list[min_idx], group_value_list[min_idx + 1])
            elif min_idx == len(group_value_list) - 1:
                chi2_list[min_idx - 1] = self.calc_chi2(count, group_value_list[min_idx - 1], group_value_list[min_idx])
                chi2_list.pop(min_idx)
            else:
                chi2_list[min_idx - 1] = self.calc_chi2(count, group_value_list[min_idx - 1], group_value_list[min_idx])
                chi2_list.pop(min_idx)
                chi2_list[min_idx] = self.calc_chi2(count, group_value_list[min_idx], group_value_list[min_idx + 1])
            # print(chi2_list)
        print("ChiMerge分箱完成！！！")

        self.group_value_list = group_value_list

        return [max(i) for i in group_value_list]


class BestDistinguishBinning(BaseBinningStrategy):
    """
    最优区分点，分箱

    最小熵 OR 最大KS

    """
    def __init__(self, label_name, distinguish_type):
        """
        :param label_name:
        :param distinguish_type: 'ks'  'entropy'
        """
        self.label_name = label_name
        self.distinguish_type = distinguish_type

    def calc(self, df, feature):

        count = pd.crosstab(df[feature], df[self.label_name])

        print(len(count.index.values.reshape(1,-1).tolist()))

        # [[]] 初始的CUT_OFF, 所有点
        cut_offs_list = [sorted(df[feature].drop_duplicates().tolist())]

        t_nums_1 = count[1].sum()

        t_nums_0 = count[0].sum()

        while len(cut_offs_list) < self.max_bin:

            cut_offs = cut_offs_list[0]

            print('----------')
            print('开始')
            print(cut_offs_list)
            print(cut_offs)

            if len(cut_offs) == 0:

                cut_offs_list.pop(0)

                continue

            # 计算每个点，当切分点时的熵
            entory_list = []

            if self.distinguish_type == 'entropy':

                entory_list = [self.calc_entropy(count[count.index <= i]) + self.calc_entropy(count[count.index > i])
                                         for i in cut_offs[:-2]]

            elif self.distinguish_type == 'ks':

                entory_list = [self.calc_ks(count[count.index <= i], t_nums_1, t_nums_0) +
                                self.calc_ks(count[count.index > i], t_nums_1, t_nums_0) for i in cut_offs[:-2]]

            min_entory_index = entory_list.index(min(entory_list))

            # 对应切分点
            cut_point = cut_offs[min_entory_index]

            print("cut_point:%s" % (cut_point))

            cut_offs_list.append(cut_offs[: min_entory_index+1])
            cut_offs_list.append(cut_offs[min_entory_index+1:])
            # 弹出已遍历过的value集合
            cut_offs_list.pop(0)

            print('结束')
            print(cut_offs_list)

        return [ele[-1] for ele in cut_offs_list] if len(cut_offs_list[0]) == 1 else\
                    [cut_offs_list[0][0]] + [ele[-1] for ele in cut_offs_list]

    @staticmethod
    def calc_entropy(count: pd.DataFrame):
        """
        计算输入数组的熵 [向量计算]
        :param count: 分组的频数统计
        :return:
        """

        # 每个取值的概率
        pi = count[1] / count.sum(axis=1)

        pi = pi[pi != 0]

        pi = - pi * np.log2(pi)

        return pi.sum()

    @staticmethod
    def calc_ks(count, t_num_1, t_num_0):
        """
        计算以idx作为分割点，分组的KS值
        :param count: DataFrame 待分箱变量各取值的正负样本数
        :param t_num_0:
        :param t_num_1:
        :return: 该分箱的ks值
        计算公式：KS_i = |sum_i / sum_T - (size_i - sum_i)/ (size_T - sum_T)|
        """

        # 计算左评分区间的累计好账户数占总好账户数比率（good %)和累计坏账户数占总坏账户数比率（bad %）。
        good_left = count[1].sum() / t_num_1 if count[1].sum() != 0 else 1
        bad_left = count[0].sum() / t_num_0 if count[0].sum() != 0 else 1

        return abs(good_left - bad_left)


if __name__ == '__main__':

    e_data = ExampleData()

    df = e_data.get_iris2()

    print(df.head())

    me_bin = BestDistinguishBinning('y', 'ks')

    me_bin.max_bin = 4

    # count = pd.crosstab(df['sepal_width'], df['y'])

    # print(MinEntropyBinning.calc_entropy(count))

    print('切分点集合', me_bin.calc(df, 'sepal_width'))