#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-25
# @Author  : HD
import pandas as pd
from analysis_evaluate.evaluate.base_evaluate import BaseEvaluate


class FeatureEvaluate(BaseEvaluate):
    """
    特征评价类
    """

    def __init__(self, df, col_label_name):
        super(BaseEvaluate, self).__init__(df, col_label_name)
