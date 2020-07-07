#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-25
# @Author  : HD
import math

from analysis_evaluate.discretization.cut_off_process import *
from constants import *
from utils.example_data import ExampleData


class BinStats(object):
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
        :return: pd.series ：key = {FEATURE，CNT，PSTV_CNT， NGTV_CNT， PRED_CNT， HIT_CNT}
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

    def calc_bin_stats(self, res):
        """
        计算分箱各组的统计指标

        :param res: pd.Series key = {FEATURE，CNT，PSTV_CNT， NGTV_CNT， PRED_CNT， HIT_CNT}
        :return: pd.Series [FEATURE, CNT, PSTV_CNT, NGTV_CNT, PRED_CNT, HIT_CNT, PRECISION, PCT, TPR, FPR, PSTV_RATE, ODDS, LN_ODDS, WOE, IV, TOTAL_IV]
        """

        if self.col_pred_name:
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
        if self.is_ascending:

            cut_offs = sorted(cut_offs)

            # Na值填充 -1
            self.df[self.col_group_name] = self.df[col_feature_name].apply(self.split_value_2box_by_cum_ascending,
                                                                           args=(cut_offs,)).fillna(-1)
        else:

            cut_offs = sorted(cut_offs, reverse=True)

            # Na值填充 - 1
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
        if value <= cut_offs[0]:

            return "(-∞, %s]" % cut_offs[0]

        for i in range(1, num_groups):

            if cut_offs[i-1] < value <= cut_offs[i]:

                return "(%s, %s]" % (cut_offs[i-1], cut_offs[i])

        # 极大异常值，归类
        return "(%s, +∞)" % cut_offs[-1]

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

            if value >= cut_offs[i]:

                return ">=%s" % cut_offs[i]

    @property
    def get_bin_stats_result_(self):
        """返回分箱结果"""
        
        return self.__bin_stats_result_df.fillna(0.0)

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

    def bin_stats_by_cum_desc(self, feature_name, percentile_list=None, cut_offs=None):
        """
        按指定百分位数或指定切分点，降序(>=value)累计，做分箱统计
        :param feature_name:
        :param percentile_list: 默认【0.9, 0.8, ..., 0】
        :param cut_offs
        :return:
        """

        self.col_feature_name = feature_name

        bbp = BinningByPercentile()

        # 指定切分点时， 返回指定点所对应的百分位数
        if cut_offs:

            """
            . 最大最小值校验
            . 保证输入数组的极大为feature.max，极小最小为feature.min
            """
            if min(cut_offs) < self.df[feature_name].min():

                cut_offs.remove(min(cut_offs))

                cut_offs.append(self.df[feature_name].min())

            if max(cut_offs) > self.df[feature_name].max():

                cut_offs.remove(max(cut_offs))

                cut_offs.append(self.df[feature_name].max())

            # 从大到最小值
            cut_offs = sorted(cut_offs, reverse=True)

            self.df['rank'] = self.df[feature_name].rank(pct=True)

            percentile_list = [self.df[self.df[feature_name] <= i]['rank'].max() for i in cut_offs]

        else:
            """
            默认十分位【10%, 20%, ..., 100%】
            """
            percentile_list = percentile_list if percentile_list else [i / 10.0 for i in range(1, 11)]

            bbp.percentile_point = percentile_list

            bbp.set_min_switch()

            cut_offs = bbp.calc(self.df, feature_name)

            # 从大到最小值
            cut_offs = sorted(cut_offs, reverse=True)

        # 各组TOPn%列
        res_dict = {TOPN_PCT_COL: [0.0] + percentile_list}

        for point in cut_offs:

            df_seg = self.df[self.df[feature_name] >= point]

            # 该分组的范围列
            res_dict.setdefault(RANGE_COL, [0.0]).append('>=%s' % point)

            # 计算该分组的基础分箱统计指标，转换为字典形式{}
            dict_seg = self.calc_base_stats_indicator(df_seg).to_dict()

            # 转换为{key: [v1, ..., vn]}
            for k, v in dict_seg.items():

                res_dict.setdefault(k, [0.0]).append(v)

        # 转换为data_frame数据结构
        res = pd.DataFrame(res_dict)

        # 计算得到， 完整的分箱统计结果
        self.__bin_stats_result_df = self.calc_bin_stats(res)

    def bin_stats_by_cum_ascd(self, feature_name, percentile_list=None, cut_offs=None):
        """
        按指定百分位数或指定切分点，累计升序（>=），进行分箱统计

        :param feature_name: str
        :param percentile_list: list
        :param cut_offs: list [极小, ..., 最大值]
        :return:
        """

        self.col_feature_name = feature_name

        bbp = BinningByPercentile()

        # 指定切分点时， 返回指定点所对应的百分位数
        if cut_offs:

            """
            . 最大最小值校验
            """
            if min(cut_offs) < self.df[feature_name].min():

                cut_offs.remove(min(cut_offs))

                cut_offs.append(self.df[feature_name].min())

            if max(cut_offs) > self.df[feature_name].max():

                cut_offs.remove(max(cut_offs))

                cut_offs.append(self.df[feature_name].max())

            # 从大到最小值
            cut_offs = sorted(cut_offs)

            self.df['rank'] = self.df[feature_name].rank(pct=True, ascending=False)

            # 获取指定各切分点的分位数
            percentile_list = [self.df[self.df[feature_name] <= i]['rank'].min() for i in cut_offs]

        # 按TopN%折算
        # 默认 【10%, 20%, ..., 100%】
        else:

            percentile_list = percentile_list if percentile_list else [i / 10.0 for i in range(1, 11)]

            bbp.percentile_point = percentile_list

            bbp.set_max_switch()

            # 输入百分位数，得到切分点的值，按<=max的形式
            cut_offs = bbp.calc(self.df, feature_name)

            # 从大到最小值
            cut_offs = sorted(cut_offs)

        # 各组TOPn%列
        res_dict = {TOPN_PCT_COL: [0.0] + percentile_list}

        for point in cut_offs:

            df_seg = self.df[self.df[feature_name] <= point]

            # 该分组的范围列
            res_dict.setdefault(RANGE_COL, [0.0]).append('<=%s' % point)

            # 计算该分组的基础分箱统计指标，转换为字典形式{}
            dict_seg = self.calc_base_stats_indicator(df_seg).to_dict()

            # 转换为{key: [v1, ..., vn]}
            for k, v in dict_seg.items():

                res_dict.setdefault(k, [0.0]).append(v)

        res = pd.DataFrame(res_dict)

        # 计算得到， 完整的分箱统计结果
        self.__bin_stats_result_df = self.calc_bin_stats(res)

    def labeled_for_feature_by_group(self, label_col_name, feature_name, cut_offs):
        """
        对指定特征列，用指定的切分点集合，按分组分箱的方式，打标签

        :param label_col_name:
        :param feature_name:
        :param cut_offs:
        :return:
        """

        self.df[label_col_name] = self.df[feature_name].apply(self.split_value_2box_by_group,
                                                                args=(cut_offs,))

    def labeled_for_feature_by_accum_desc(self, label_col_name, feature_name, cut_offs):
        """
        对指定特征列，用指定的切分点集合，按降序累计的分箱方式，打标签

        :param label_col_name:
        :param feature_name:
        :param cut_offs:
        :return:
        """

        self.df[label_col_name] = self.df[feature_name].apply(self.split_value_2box_by_cum_descending,
                                                                args=(cut_offs,))

    def labeled_for_feature_by_accum_ascd(self, label_col_name, feature_name, cut_offs):
        """
        对指定特征列，用指定的切分点集合，按升序累计的分箱方式，打标签

        :param label_col_name:
        :param feature_name:
        :param cut_offs:
        :return:
        """

        self.df[label_col_name] = self.df[feature_name].apply(self.split_value_2box_by_cum_ascending,
                                                                args=(cut_offs,))

    def bin_stats_by_group(self, feature_name, cut_offs):
        """
        按指定切分点， 分组分箱
        :param feature_name:
        :param cut_offs:
        :return:
        """

        self.col_feature_name = feature_name

        # 按左开右闭，进行分组标记
        self.df['group'] = self.df[feature_name].apply(self.split_value_2box_by_group,
                                                         args=(cut_offs,))

        # 按分组标记， 计算基础各分箱的基础统计指标
        res = self.df.groupby('group').apply(self.calc_base_stats_indicator)

        # 计算分箱统计指标
        res = self.calc_bin_stats(res)

        # 重置index，让df扁平
        res.reset_index(inplace=True)

        # 第一列的名称，’group' -> RANGE_COL
        res.rename(columns={res.columns.values[0]: RANGE_COL}, inplace=True)

        self.__bin_stats_result_df = res

    def bin_stats_by_monotone_test(self, col_feature_name, cut_off_strategy: BaseBinningStrategy, init_max_bin=10):
        """
        按指定策略， 通過減少max_bin，來搜索单调分区
        :param col_feature_name: 待計算的特征列名
        :param cut_off_strategy: 分項策略
        :param init_max_bin: 初始最大箱数
        :return:
        """

        self.__is_accumulate_bin = False

        # 分组分箱
        self.__is_group_bin = True

        # 是否单调。 跳出while循环的标记
        monotonic_ok = False

        cut_off_strategy.max_bin = init_max_bin

        """
        减小max_bin，直到分区结果单调
        """
        while not monotonic_ok:

            cut_off_strategy.max_bin = init_max_bin

            # 计算切分点集合
            cut_offs = cut_off_strategy.calc(self.df, col_feature_name)

            print('-------')
            print(cut_offs)

            # 当前最大分箱数，进行分组分箱
            self.bin_stats_by_group(col_feature_name, cut_offs)

            # 每个分区的正样本比例
            pstv_ratio_df = self.__bin_stats_result_df[PSTV_RATE_COL]

            # 更新状态， 单调递增 或者 单调递减
            monotonic_ok = pstv_ratio_df.is_monotonic_decreasing or pstv_ratio_df.is_monotonic_increasing

            init_max_bin -= 1

            print(self.get_bin_stats_result_[[RANGE_COL, CNT_COL, PSTV_RATE_COL]])

    def bin_error_(self):

        P = 0.05

        # 获取一列数得Q1
        get_quantile_25 = lambda x: x.quantile(P)

        # 获取一列数得Q3
        get_quantile_75 = lambda x: x.quantile(1 - P)

        # 匿名函数重命名
        get_quantile_25.__name__ = 'quantile_25'

        get_quantile_75.__name__ = 'quantile_75'

        self.df.groupby(self.label_name)[['f1', 'f2']].agg({'f1': 'mean', 'f2': [get_quantile_25, get_quantile_75]})


if __name__ == '__main__':

    e_data = ExampleData()

    df = e_data.get_iris2()

    print(df.head())

    be = BinStats(df, 'y')

    df['grp'] = be.split_value_2box_by_group(1, [4,5,6,7])

    # 卡方切分策略
    chi2_strategy = Chi2Binning('y')

    # 等频切分策略
    ef_strategy = EqualFrequencyBinning()

    # 等距切分策略
    ew_strategy = EqualWidthBinning()

    # 指定分位数
    eff_strategy = BinningByPercentile()

    # eff_strategy.percentile_point = [1,0.9,0.8,0.6,0.7,0.5,0.4,0.3,0.2,0.1]

    # be.bin_stats_by_group('sepal_width', ef_strategy.calc(df, 'sepal_width'))

    be.bin_stats_by_monotone_test('sepal_width', ef_strategy)

    # be.bin_stats_by_cum_desc('sepal_width', cut_offs=[1.9, 3.2, 8])

    be.bin_stats_by_cum_ascd('sepal_width')

    print(be.get_bin_stats_result_[[TOPN_PCT_COL, RANGE_COL, CNT_COL, TPR_COL, PSTV_RATE_COL]])

    # print(chi2_strategy.group_value_list)

    # be.bin_by_manual('sepal_width', [2.8, 3.1, 4.4])

    # print(be.get_bin_stats_result_[[RANGE_COL, CNT_COL, TPR_COL, PSTV_RATE_COL]])

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