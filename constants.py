#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-25
# @Author  : HD


KEY = 'A'

"""
分箱类中的列名
"""
FEATURE_COL = 'Var'  # 列名
RANGE_COL = 'range'  # 取值范围
TOPN_PCT_COL = 'topN%'  # 取值前N%
MIN_COL = 'min'
MAX_COL = 'max'
CNT_COL = 'cnt'  # 该组总数
CUM_CNT_COL = 'cum_cnt'  # 累计总数
NGTV_CNT_COL = 'ngtv_cnt'  # 该组负样本
PSTV_CNT_COL = 'pstv_cnt'  # 该组正样本
CUM_NGTV_CNT_COL = 'ngtv_cnt'  # 该组负样本
CUM_PSTV_CNT_COL = 'pstv_cnt'  # 该组正样本
PCT_COL = 'pct'  # 该组样本所占比例
TPR_COL = 'tpr'  # 该组TPR
FPR_COL = 'fpr'  # 该组FPR
PSTV_RATE_COL = 'pstv_rate'  # 该组的正样本比例 = 该组正样本数 / 该组样本数
ODDS_COL = 'odds'  # 该组正样本比例 / 该组负样本比例
LN_ODDS_COL = 'lnodds'  # ODD取log
WOE_COL = 'woe'
IV_COL = 'iv'
TOTAL_IV_COL = 't_iv'
PRED_CNT_COL = 'pred_cnt'  # 预测1样本的个数
HIT_CNT_COL = 'hit_cnt'  # 命中个数
PRECISION_COL = 'precision'  # 预测精度