#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-29
# @Author  : HD
from analysis_evaluate.discretization.cut_off_process import EqualFrequencyBinning, BinningByPercentile
from analysis_evaluate.discretization.bin_stats import BinStats
from analysis_evaluate.my_plot import MyPlot
from utils.constants import *
import pandas as pd
import numpy as np

from utils.example_data import ExampleData


class BaseEvaluate(object):
    """
    评价基础类
    """

    # Y标签列名
    col_label_name = ''

    # 待分箱的特征名称
    feature_name = ''

    # 特征切分点数组
    cut_off_list = []

    def __init__(self, df, col_label_name):
        super(object, self).__init__()
        self.df = df
        self.col_label_name = col_label_name
        self.bin_stats = BinStats(df, col_label_name)
        self.my_plot = MyPlot()

    def ks_curve(self):
        """
        基于降序累计的分箱统计结果， 画KS图。

        :return:
        """

        ks_score = max(self.bin_stats.get_bin_stats_result_[TPR_COL] - self.bin_stats.get_bin_stats_result_[FPR_COL])

        self.my_plot.title = 'KS-CURVE'

        self.my_plot.add_figure_text(0.02, 0.65, 'ks_score=%.2f' % ks_score)

        self.my_plot.plot_by_x_group_y(self.bin_stats.get_bin_stats_result_[TOPN_PCT_COL],
                                       {'tpr': self.bin_stats.get_bin_stats_result_[TPR_COL],
                                        'fpr': self.bin_stats.get_bin_stats_result_[FPR_COL]})

    def gain_curve(self):
        """
        基于降序累计，增益曲线
        """

        x_list = self.bin_stats.get_bin_stats_result_[TOPN_PCT_COL]

        # 基线
        cum_tpr_base_line = x_list

        self.my_plot.title = 'GainPlot'

        self.my_plot.x_label = 'topRatio%Population'

        # 概率的gains增益图[累计捕获]
        self.my_plot.plot_by_x_group_y(x_list,
                                       {'tpr': self.bin_stats.get_bin_stats_result_[TPR_COL],
                                        'base': cum_tpr_base_line})

    def kde_curve_by_group_col(self, feature_name, group_name):

        self.my_plot.title = '%s_kde_plot' % feature_name

        self.my_plot.x_lim = [0, 1]

        self.my_plot.kde_plot_by_group(self.df, feature_name, group_name)

    def plot_woe(self):

        woe = self.bin_stats.get_bin_stats_result_[[RANGE_COL, WOE_COL]]

        mp = MyPlot()

        mp.title = 'WOE'

        mp.bar_plot(woe, WOE_COL, RANGE_COL)

    @property
    def gini_coefficient_(self):
        """
        基于升序累计， 计算基尼系数

        Wi = 第1组，累计到最后一组的样本的总正样本数（收入）占总体正样本数（总收入）的比例

        1-1 / D16 * (2*SUM(D5:D13)+1)

        :return:
        """
        return 1 - 1/len(self.bin_stats.get_bin_stats_result_) * (2 * self.bin_stats.get_bin_stats_result_[TPR_COL].sum() + 1)

    @property
    def gini_coefficient2_(self):
        """基于升序累计， 计算基尼系数"""

        y_array = self.bin_stats.get_bin_stats_result_['tpr'].tolist()

        x_array = np.array(range(0, len(y_array))) / np.float(len(y_array) - 1)

        B = np.trapz(y_array, x=x_array)

        A = 0.5 - B

        return A / (A + B)


if __name__ == '__main__':

    e_data = ExampleData()

    df = e_data.get_iris2()

    print(df.head())

    be = BaseEvaluate(df, 'y')

    ef = EqualFrequencyBinning()

    ef.max_bin = 5

    # be.bin_stats.bin_stats_by_group('sepal_width', ef.calc(df, 'sepal_width'))

    be.bin_stats.bin_stats_by_monotone_test('sepal_width', ef)

    print(be.bin_stats.get_bin_stats_result_[[RANGE_COL, CNT_COL, PSTV_CNT_COL, TPR_COL]])

    # print(be.gini_coefficient_)

    # print(be.gini_coefficient2_)

    # print(be.bin_stats.get_bin_stats_result_)
    #
    # print(be.bin_stats.get_bin_stats_result_[[RANGE_COL, PSTV_RATE_COL]])
    #
    # be.ks_curve()
    # #
    # be.gain_curve()
    # #
    # be.kde_curve_by_group_col('sepal_width', 'species')
    # #
    # be.plot_woe()
    import platform

    platform.platform()

