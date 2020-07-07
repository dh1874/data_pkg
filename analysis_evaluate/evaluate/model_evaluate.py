#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-25
# @Author  : HD
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from analysis_evaluate.discretization.bin_stats import BinStats
from constants import *
from analysis_evaluate.evaluate.base_evaluate import BaseEvaluate


class ModelEvaluate(BaseEvaluate):
    """
    模型评价类
    """

    def __init__(self, df, col_label_name, col_pred_label_name, col_prob_name):
        super(BaseEvaluate, self).__init__(df, col_label_name)
        self.bin_stats = BinStats(df, col_label_name, col_pred_label_name)
        self.col_pred_label_name = col_pred_label_name
        self.col_prob_name = col_prob_name
        # 按默认【0.9，0.8，...，0】的分位数，降序累计，计算概率值的分统计结果
        self.bin_stats.bin_stats_by_cum_desc(self.col_prob_name)

    def roc_curve(self):
        """
        画ROC曲线
        :return:
        """

        auc_score = roc_auc_score(self.df[self.col_label_name], self.df[self.col_prob_name])

        self.my_plot.title = 'ROC-CURVE'

        self.my_plot.x_label = 'False_Positive_Rate'

        self.my_plot.y_label = 'True_Positive_Rate'

        self.my_plot.add_figure_text(0.02, 0.98, 'auc=%s' % auc_score)

        self.my_plot.plot_by_xy(self.bin_stats.get_bin_stats_result_[FPR_COL],
                                self.bin_stats.get_bin_stats_result_[TPR_COL])

    def lift_curve(self):

        # 1样本，在总体中的比例
        pi1 = self.bin_stats.t_positive_cnt / self.bin_stats.t_cnt

        # 模型Lift值集合
        # 每个分组中正样本比例 / 整体中正样本的比例
        lift_model = [i / pi1 for i in self.bin_stats.get_bin_stats_result_[PSTV_RATE_COL]][1:]

        # 基线
        lift_base_line = [1] * 10

        self.my_plot.title = 'LiftPlot'

        self.my_plot.x_label = 'topRatio%Population'

        self.my_plot.plot_by_xy(self.bin_stats.get_bin_stats_result_[TOPN_PCT_COL],
                                {'model': lift_model, 'base': lift_base_line})

    def bep_plot(self):
        """
        平衡点图（precision和recall的平衡）

        :return:
        """

        # 基线
        bep_base_line = [i / 10.0 for i in range(11)]

        self.my_plot.title = 'BepPlot'

        self.my_plot.x_label = 'topRatio%Population'

        # 作图
        self.my_plot.plot_by_x_group_y(self.bin_stats.get_bin_stats_result_[TOPN_PCT_COL],
                                      {'precision': self.bin_stats.get_bin_stats_result_[PRECISION_COL],
                                       'recall': self.bin_stats.get_bin_stats_result_[TPR_COL],
                                       'base': bep_base_line})

    def probability_kde_plot(self):
        """
        正负样本的核密度曲线对比图示
        :return:
        """

        self.kde_curve_by_group_col(self.col_prob_name, self.col_pred_label_name)

    def metric(self):
        """
        混淆矩阵

        :return:
        """

        metric_rst = pd.crosstab(self.df[self.col_label_name],
                                 self.df[self.col_pred_label_name],
                                 rownames=['actual'],
                                 colnames=['predictions'])

        print(metric_rst)

        print(classification_report(self.df[self.col_label_name], self.df[self.col_pred_label_name]))


if __name__ == '__main__':

    df = 0

    me = ModelEvaluate(df, 'y', 'pred', 'prob')
