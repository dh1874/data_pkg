#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-25
# @Author  : HD
import pandas as pd
from analysis_evaluate.evaluate.base_evaluate import BaseEvaluate
from utils.example_data import ExampleData


class FeatureEvaluate(BaseEvaluate):
    """
    特征评价类，分类
    """

    def __init__(self, df, col_label_name):
        """
        :param df: 待评价的df
        :param col_label_name: 标签列名
        """
        super(BaseEvaluate, self).__init__(df, col_label_name)


if __name__ == '__main__':

    e_data = ExampleData()

    df = e_data.get_iris2()

    print(df.head())

    fe = FeatureEvaluate(df, 'y')

    fe.kde_curve_by_group_col('sepal_width',)

    # 对指定特征， 按某种方式分箱
    fe.bin_stats.bin_stats_by_cum_desc('sepal_width')

    fe.ks_curve()
