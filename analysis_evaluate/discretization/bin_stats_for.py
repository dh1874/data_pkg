#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-25
# @Author  : HD
import math

from analysis_evaluate.discretization.cut_off_process import *
from constants import *
from utils.example_data import ExampleData


class BinStatsFor(object):
    """
    分箱统计
    """

    # 总样本数
    t_nbr_samples = 0

    # 样本数
    nbr_samples = 0

    # 正样本数
    nbr_positive_samples = 0

    # 负样本数
    nbr_negative_samples = 0

    # 预测为正的样本数
    nbr_pred_positive = 0

    tp = 0

    # 预测为负的样本数
    nbr_pred_negative = 0

    tn = 0

    # precision
    precision = 0

    # accuracy
    accuracy = 0

    # recall TPR
    tpr = 0

    # fpr
    fpr = 0

    # f值
    f_score = 0

    # 正樣本占比
    positive_samples_percentage = 0.0

    # 負樣本占总体的分布比例
    distribution_negative = 0.0

    # 正样本占总体的分布比例
    distribution_positive = 0.0

    # WOE值
    woe = 0.0

    # 正样本标签值
    POSITIVE_LABEL_VALUE = 1.0

    # 负样本标签纸
    NEGATIVE_LABEL_VALUE = 0.0

    col_feature_name = ''

    col_group_name = ''

    # 分箱结果
    __bin_stats_result_df: pd.DataFrame

    # 是否，切分点累积的方式，计算分箱的方式
    __is_accumulate_bin = False

    # 切分点分组，分箱
    __is_group_bin = False

    # 默认升序排列
    is_ascending = True

    # 是否，分箱的單調性检验
    is_monotone_test = False

    # 使用的CUT_OFFS
    _cut_offs = []

    def __init__(self, df, label_name, col_pred_name=False):
        self.df: pd.DataFrame = df
        self.label_name = label_name
        self.t_cnt = len(df)
        self.t_positive_cnt = df[label_name].sum()
        self.t_negative_cnt = self.t_cnt - self.t_positive_cnt
        self.col_pred_name = col_pred_name  # 模型预测值列名

    def calc_base_stats_indicator(self, data_series):
        """
        计算基础的分箱统计指标

        :param data_series:
        :return: pd.series
        """

        rst_dict = {}

        nbr_samples = len(data_series)

        # y = 1 的人数
        nbr_positive_samples = data_series[self.label_name].sum()

        # y = 0 的人数
        nbr_negative_samples = nbr_samples - nbr_positive_samples

        # maxs = data_series[self.col_feature_name].max()
        #
        # mins = data_series[self.col_feature_name].min()

        rst_dict.update({FEATURE_COL: self.col_feature_name,
                         CNT_COL: nbr_samples,
                         # MIN_COL: mins,
                         # MAX_COL: maxs,
                         PSTV_CNT_COL: nbr_positive_samples,
                         NGTV_CNT_COL: nbr_negative_samples})

        if self.col_pred_name:

            nbr_pred_positive = df[self.col_pred_name].sum()

            hit_cnt = len(df[df[self.col_pred_name] == df[self.label_name]])

            rst_dict = {PRED_CNT_COL: nbr_pred_positive,
                        HIT_CNT_COL: hit_cnt}

        return pd.Series(rst_dict)

    def calc_bin_stats_by_group(self, col_group_name):
        """
        计算分箱各组的统计指标

        :param col_group_name:
        :return: {TPR, FPR}
        """

        res: pd.DataFrame = self.df.groupby(self.df[col_group_name]).apply(
                                    self.calc_base_stats_indicator)

        if self.__is_accumulate_bin:

            res[CNT_COL] = res[CNT_COL].cumsum()
            res[PSTV_CNT_COL] = res[PSTV_CNT_COL].cumsum()
            res[NGTV_CNT_COL] = res[NGTV_CNT_COL].cumsum()

            if self.col_pred_name:
                res[PRED_CNT_COL] = res[PRED_CNT_COL].cumsum()
                res[HIT_CNT_COL] = res[HIT_CNT_COL].cumsum()
                res[PRECISION_COL] = res[HIT_CNT_COL] / res[PRED_CNT_COL]

        res[PCT_COL] = res[CNT_COL] / self.t_cnt
        res[TPR_COL] = res[PSTV_CNT_COL] / self.t_positive_cnt
        res[FPR_COL] = res[NGTV_CNT_COL] / self.t_negative_cnt
        res[PSTV_RATE_COL] = res[PSTV_CNT_COL] / res[CNT_COL]
        res[ODDS_COL] = res[PSTV_RATE_COL] / (1 - res[PSTV_RATE_COL])
        res[LN_ODDS_COL] = np.log(res[ODDS_COL])
        res[WOE_COL] = np.log(res[TPR_COL] / res[FPR_COL])
        res[IV_COL] = (res[TPR_COL] - res[FPR_COL]) * res[WOE_COL]
        res[TOTAL_IV_COL] = res[IV_COL].replace({np.inf: 0, -np.inf: 0}).sum()

        res.reset_index(inplace=True)

        res.rename(columns={res.columns.values[0]: RANGE_COL}, inplace=True)

        return res

    def calc_psi(self, df1, df2, col_name):
        """
        计算群体稳定性指标

        PSI = SUM( (实际占比 - 预期占比）* ln(实际占比 / 预期占比) )

        :param df1:
        :param df2:
        :param col_name: 待考察的列名
        :return:
        """
        pass

    def calc_woe_iv(self, nums_1, nums_0, t_nums_1, t_nums_0):
        """
        计算全量样本中，某一分组的 WOE值，和IV_值

        :param nums_1:  该分组的1样本数（real)
        :param nums_0:  该分组的0样本数（real)
        :param t_nums_1:  总体1样本数
        :param t_nums_0:  总体0样本数
        :return: {woe:, iv:}
        """

        # 数值修正 【避免IV的极端值】
        # 该组1 OR 0样本为0时， 调整为 1
        tpr = nums_1 / t_nums_1 if nums_1 > 0 else 1 / t_nums_1

        fpr = nums_0 / t_nums_0 if nums_0 > 0 else 1 / t_nums_0

        woe = round(math.log(tpr / fpr), 2)

        iv = round(woe * (tpr - fpr), 2)

        return {WOE_COL: woe, IV_COL: iv}

    def calc_group_label_by_bins(self, col_feature_name, cut_offs):
        """
        分箱分组， 生成新列

        :param col_feature_name: 待考察的特征列名
        :param cut_offs:
        :return: self.df
        """

        self.col_group_name = 'grp_by_' + col_feature_name

        # 升序累计 <=min, ..., <=max
        if self.is_ascending and self.__is_accumulate_bin:

            cut_offs = sorted(cut_offs)

            # Na值填充 -1
            self.df[self.col_group_name] = self.df[col_feature_name].apply(self.split_value_2box_by_cum_ascending,
                                                                           args=(cut_offs,)).fillna(-1)
        else:

            cut_offs = sorted(cut_offs, reverse=True)

            # Na值填充 -1
            self.df[self.col_group_name] = self.df[col_feature_name].apply(self.split_value_2box_by_cum_descending,
                                                                           args=(cut_offs,)).fillna(-1)

    @staticmethod
    def split_value_2box_by_group(value, cut_offs):
        """
        切分点转group列名, 通过分组

        :param value:
        :param cut_offs: list
        :return:
        """

        # 升序
        # cut_offs = sorted(cut_offs)
        num_groups = len(cut_offs)

        # 极小异常值， 归类
        if value < cut_offs[0]:

            return "<%s" % cut_offs[0]

        for i in range(1, num_groups):

            if cut_offs[i-1] <= value < cut_offs[i]:

                return "[%s, %s)" % (cut_offs[i-1], cut_offs[i])

        # 极大异常值，归类
        return ">=%s" % cut_offs[-1]

    @staticmethod
    def split_value_2box_by_cum_ascending(value, cut_offs):
        """
        切分点转group列名, 通过<=方式

        :param value:
        :param cut_offs: list【min, ..., max】
        :return:
        """

        # 升序
        # cut_offs = sorted(cut_offs)

        num_groups = len(cut_offs)

        for i in range(0, num_groups):

            if value <= cut_offs[i]:

                return "<=%s" % cut_offs[i]

    @staticmethod
    def split_value_2box_by_cum_descending(value, cut_offs):
        """
        切分点转group列名, 通过>=方式

        :param value:
        :param cut_offs: list【maxs, ..., min】
        :return:
        """

        # 升序
        # cut_offs = sorted(cut_offs)

        num_groups = len(cut_offs)

        for i in range(0, num_groups):

            if value >= cut_offs[num_groups - 1 - i]:

                return ">=%s" % cut_offs[i]

    @property
    def get_bin_stats_result_(self):
        """返回分箱结果"""
        
        return self.__bin_stats_result_df

    @property
    def iv_(self):
        """
        返回当前feature，分箱计算后的IV
        :return:
        """
        return self.__bin_stats_result_df[TOTAL_IV_COL].values[0]

    @property
    def is_accumulate_bin(self):
        return self.__is_accumulate_bin

    @is_accumulate_bin.setter
    def is_accumulate_bin(self, value):
        self.__is_accumulate_bin = value
        self.__is_group_bin = False

    @property
    def is_group_bin(self):
        return self.__is_accumulate_bin

    @is_group_bin.setter
    def is_group_bin(self, value):
        self.__is_group_bin = value
        self.__is_accumulate_bin = False

    def bin_by_manual(self, col_feature_name, cut_offs):
        """
        按给定的切分点集合， 对指定特征， 进行分箱统计操作

        :param col_feature_name:
        :param cut_offs:
        :return:
        """

        res: pd.DataFrame = self.df.groupby(self.df[col_group_name]).apply(
                                    self.calc_base_stats_indicator)

        if self.__is_accumulate_bin:

            res[CNT_COL] = res[CNT_COL].cumsum()
            res[PSTV_CNT_COL] = res[PSTV_CNT_COL].cumsum()
            res[NGTV_CNT_COL] = res[NGTV_CNT_COL].cumsum()

            if self.col_pred_name:
                res[PRED_CNT_COL] = res[PRED_CNT_COL].cumsum()
                res[HIT_CNT_COL] = res[HIT_CNT_COL].cumsum()
                res[PRECISION_COL] = res[HIT_CNT_COL] / res[PRED_CNT_COL]

        res[PCT_COL] = res[CNT_COL] / self.t_cnt
        res[TPR_COL] = res[PSTV_CNT_COL] / self.t_positive_cnt
        res[FPR_COL] = res[NGTV_CNT_COL] / self.t_negative_cnt
        res[PSTV_RATE_COL] = res[PSTV_CNT_COL] / res[CNT_COL]
        res[ODDS_COL] = res[PSTV_RATE_COL] / (1 - res[PSTV_RATE_COL])
        res[LN_ODDS_COL] = np.log(res[ODDS_COL])
        res[WOE_COL] = np.log(res[TPR_COL] / res[FPR_COL])
        res[IV_COL] = (res[TPR_COL] - res[FPR_COL]) * res[WOE_COL]
        res[TOTAL_IV_COL] = res[IV_COL].replace({np.inf: 0, -np.inf: 0}).sum()

        res.reset_index(inplace=True)

        res.rename(columns={res.columns.values[0]: RANGE_COL}, inplace=True)

        self.__bin_stats_result_df = res

    def bin_by_strategy(self, col_feature_name, cut_off_strategy: BaseBinningStrategy):
        """
        按指定策略，生成切分點集合, 計算分箱結果

        :param col_feature_name: 指定特征
        :param cut_off_strategy: 指定策略
        :return:
        """

        cut_offs = cut_off_strategy.calc(self.df, col_feature_name)

        # 单调校验
        if self.is_monotone_test:

            self.__is_accumulate_bin = False

            self.__is_group_bin = True

            monotonic_ok = False

            max_bin = cut_off_strategy.max_bin

            while not monotonic_ok:

                cut_off_strategy.max_bin = max_bin

                cut_offs = cut_off_strategy.calc(self.df, col_feature_name)

                print('-------')
                print(cut_offs)

                self.bin_by_manual(col_feature_name, cut_offs)

                pstv_ratio_df = self.__bin_stats_result_df[PSTV_RATE_COL]

                # 单调递增 或者 单调递减
                monotonic_ok = pstv_ratio_df.is_monotonic_decreasing or pstv_ratio_df.is_monotonic_increasing

                max_bin -= 1

                print(self.get_bin_stats_result_[[RANGE_COL, CNT_COL, PSTV_RATE_COL]])

        else:

            self.bin_by_manual(col_feature_name, cut_offs)


if __name__ == '__main__':

    e_data = ExampleData()

    df = e_data.get_iris2()

    print(df.head())

    be = BinStats(df, 'y')

    # 卡方切分策略
    chi2_strategy = Chi2Binning('y')

    # 等频切分策略
    ef_strategy = EqualFrequencyBinning()

    # 等距切分策略
    ew_strategy = EqualWidthBinning()

    # 单调校验
    # be.is_monotone_test = True

    be.is_accumulate_bin = True

    # be.is_ascending = False

    # 按策略分箱统计
    be.bin_by_strategy('sepal_width', ef_strategy)

    # print(chi2_strategy.group_value_list)

    # be.is_group_bin = True

    # be.bin_by_manual('sepal_width', [2.8, 3.1, 4.4])

    print(be.get_bin_stats_result_[[RANGE_COL, CNT_COL, PSTV_RATE_COL]])

    """
         Range   BadRate
    0  <=2.800  0.404255
    1  <=3.100  0.382979
    2  <=4.400  0.232143
    """

    # print(be.get_bin_stats_result_[[RANGE_COL, PSTV_RATE_COL]])

    # cut_offs = be.calc_cut_offs_by_k_means('sepal_length', 3)

    # cut_offs = be.calc_cut_offs_by_chi2('sepal_length', bins=5)
    #
    # print(cut_offs)
    #
    # be.manual_bin(cut_offs, 'sepal_length')

    # be.calc_group_label_by_bins('sepal_width', [1, 2, 3], is_group=True)

    # print(be.df.head())