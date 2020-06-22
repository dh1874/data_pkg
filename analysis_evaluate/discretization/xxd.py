#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-28
# @Author  : 徐小东(xxd626@outlook.com) and 郁晴(873237045@qq.com)


from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from scipy.stats import chi2

from analysis_evaluate.my_plot import MyPlot
from utils.example_data import ExampleData

plt.style.use('seaborn')
import pandas as pd
import numpy as np
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


class XXDNumberBin(object):
    def __init__(self):
        self.__bin_stats = None

    def get_bin_stats(self):
        if self.__bin_stats is not None:
            return self.__bin_stats.reset_index(drop=True)

    def get_cutoff(self):
        if self.__bin_stats is not None:
            return self.__bin_stats.Max.dropna().tolist()

    def trans_bin_to_woe(self, B):
        '''

        B: Series
        '''
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        woe = self.__bin_stats['WoE'].sort_index()
        return B.map(lambda x: woe[x])

    def plot_woe(self, title=None):
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        woe = self.__bin_stats[['WoE', 'Range']].sort_index()
        plt.clf()
        if title is None:
            title = self.__varname
        plt.title('{}(WOE)'.format(title))
        plt.bar(range(len(woe)), woe.WoE, tick_label=woe.Range)
        plt.show()
        print('Cutoff:{}'.format(self.get_cutoff()))

        mp = MyPlot()

        mp.title('woe')

        mp.bar_plot(self.__bin_stats, 'woe', 'range')

    def get_iv(self):
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        return self.__bin_stats['TotalIV'].iloc[0]

    def get_varname(self):
        return self.__varname

    def trans_to_bin(self, X):
        '''
        如果训练集有缺失：
        1）缺失值分到缺失组，
        2）小于最小值的分到第一组
        3) 超过最大值的分最后一组。
        如果训练集没有缺失：
        1）缺失值\小于最小值分到第一组；
        2）超过最大值的分最后一组

        X: series
        '''
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        if not is_numeric_dtype(X):
            X = X.astype(float)
        cuts = self.__bin_stats['Max'].sort_values(na_position='first')
        mx = cuts.max()
        return X.map(lambda x: (cuts >= x).idxmax() if x <= mx else cuts.index[-1], na_action='ignore').fillna(
            cuts.index[0])

    def trans_to_woe(self, X):
        '''
        如果训练集有缺失：
        1）缺失值分到缺失组，
        2）小于最小值的分到第一组
        3) 超过最大值的分最后一组。
        如果训练集没有缺失：
        1）缺失值\小于最小值分到第一组；
        2）超过最大值的分最后一组
        X : series
        '''
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        if not is_numeric_dtype(X):
            X = X.astype(float)
        cuts = self.__bin_stats['Max'].sort_values(na_position='first')
        mx = cuts.max()
        woe = self.__bin_stats['WoE'].sort_index()
        return X.map(lambda x: woe[(cuts >= x).idxmax()] if x <= mx else woe.iloc[-1], na_action='ignore').fillna(
            woe.iloc[0])

    def __cc(self, dfx):
        mx = dfx.XX.max()
        mn = dfx.XX.min()
        cnt = len(dfx)
        bad = dfx.YY.sum()
        good = cnt - bad
        return pd.Series({'Var': self.__varname, 'Range': '<={:.3f}'.format(mx) if pd.notnull(mx) else 'Miss',
                          'Min': mn, 'Max': mx, 'CntRec': cnt, 'CntGood': good, 'CntBad': bad})

    def calc_stats(self, data):
        '''
        计算woe，iv等。
        data: df[['bin','XX',YY']]
        '''
        res = data.groupby(data['bin']).apply(self.__cc)
        cntg = (data.YY == 0).sum()
        cntb = (data.YY == 1).sum()
        res['Pct'] = res.CntRec / len(data)
        res['PctBad'] = res.CntBad / cntb
        res['PctGood'] = res.CntGood / cntg
        res['BadRate'] = res.CntBad / res.CntRec
        res['CumGood'] = res.CntGood.cumsum()
        res['CumBad'] = res.CntBad.cumsum()
        res['Odds'] = res.BadRate / (1 - res.BadRate)
        res['LnOdds'] = np.log(res.Odds)
        res['WoE'] = np.log(res.PctBad / res.PctGood)
        res['IV'] = (res.PctBad - res.PctGood) * res.WoE
        res['TotalIV'] = res.IV.replace({np.inf: 0, -np.inf: 0}).sum()
        # res=res.append(pd.Series({'Var':x,'Min':XX.min(),'Max':XX.max(),'LnOdds':np.log(),'IV':res.IV.sum()},name='ALL'))
        return res

    def manual_bin(self, df, x, y, cutoff=[]):
        '''
        手动分箱
        df: 数据
        x: 变量名
        y: 目标变量
        '''
        self.__varname = x
        XX, YY = df[x], df[y]
        assert YY.isin([0, 1]).all(), 'ERROR: {} 目标变量非0/1!'.format(y)
        if not is_numeric_dtype(XX):
            XX = XX.astype(float)
        data = pd.DataFrame({'XX': XX, 'YY': YY})
        cnt = XX.count()
        assert cnt > 0, 'ERROR: "{}" 变量值全为 NULL  !'.format(x)
        edges = pd.Series(cutoff + [np.inf]).sort_values()
        mx = edges.max()
        data['bin'] = XX.map(lambda x: (edges >= x).idxmax() if x <= mx else edges.index[-1],
                             na_action='ignore').fillna(-1)
        self.__bin_stats = self.calc_stats(data)

    def pct_bin(self, df, x, y, max_bin=10, min_pct=0.06):
        '''
        等频分箱。
        df: 数据
        x: 变量名
        y: 目标变量
        '''
        self.__varname = x
        XX, YY = df[x], df[y]
        assert YY.isin([0, 1]).all(), 'ERROR: {}  目标变量非0/1!'.format(y)
        if not is_numeric_dtype(XX):
            XX = XX.astype(float)
        data = pd.DataFrame({'XX': XX, 'YY': YY})
        cnt = XX.count()
        assert cnt > 0, 'ERROR: "{}" 变量值全为 NULL  !'.format(x)
        min_sample = int(len(XX) * min_pct)
        if cnt <= min_sample:
            print('WARN: "{}" 非空值少于 {} !'.format(x, min_pct))
        nuniq = XX.nunique()
        if nuniq <= 50:
            print('WARN: "{}" 数值型变量只有 {} 个取值!'.format(x, nuniq))
        cut_ok = False
        ZZ = XX.rank(pct=1)
        while not cut_ok:
            edges = pd.Series(np.linspace(0, 1, max_bin + 1))
            bins = ZZ.map(lambda r: (edges >= r).idxmax(), na_action='ignore').fillna(-1)
            cut_ok = True
            if bins.value_counts().min() < min_sample and cnt > min_sample and max_bin > 1:
                max_bin = max_bin - 1
                cut_ok = False
        data['bin'] = bins
        self.__bin_stats = self.calc_stats(data)

    def monotone_bin(self, df, x, y, max_bin=10):
        '''
        单调分箱。
        df: 数据
        x: 变量名
        y: 目标变量
        '''
        self.__varname = x
        XX, YY = df[x], df[y]
        assert YY.isin([0, 1]).all(), 'ERROR: {} 目标变量非0/1!'.format(y)
        if not is_numeric_dtype(XX):
            XX = XX.astype(float)
        data = pd.DataFrame({'XX': XX, 'YY': YY})
        cnt = XX.count()
        assert cnt > 0, 'ERROR: "{}" 变量值全为 NULL  !'.format(x)
        cut_ok = False
        ZZ = XX.rank(pct=1)  # 排序名次，按百分比展示。 該值為TOPX-pct位置。
        while not cut_ok:
            edges = pd.Series(np.linspace(0, 1, max_bin + 1))
            data['bin'] = ZZ.map(lambda r: (edges >= r).idxmax(), na_action='ignore').fillna(-1)
            res = self.calc_stats(data).sort_index()
            woe = res[~res.Max.isna()].WoE
            cut_ok = woe.is_monotonic_decreasing or woe.is_monotonic_increasing
            max_bin = max_bin - 1
        self.__bin_stats = res


class XXDCharBin():
    def __init__(self):
        self.__bin_stats = None

    def get_bin_stats(self):
        if self.__bin_stats is not None:
            return self.__bin_stats.copy()

    def trans_bin_to_woe(self, B):
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        data = B.to_frame()
        woe = self.__bin_stats['WoE'].sort_index()
        return B.map(lambda x: woe[x], na_action='ignore').fillna(woe.iloc[0])

    def plot_woe(self, title=None):
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        woe = self.__bin_stats[['WoE', 'Range']].sort_values(by='WoE')
        plt.clf()
        if title is None:
            title = self.__varname
        plt.title('{}(WOE)'.format(title))
        plt.bar(range(len(woe)), woe.WoE)
        plt.show()
        print(woe.Range.reset_index(drop=True))

    def get_iv(self):
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        return self.__bin_stats['TotalIV'].iloc[0]

    def get_varname(self):
        return self.__varname;

    def trans_to_bin(self, X):
        '''
        新值分到缺失
        X: series
        '''
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        if not is_string_dtype(X):
            X = X.astype(str)
        data = X.to_frame()
        data['bin'] = -1
        for bin, values in enumerate(self.__bins):
            data.loc[X.isin(values), 'bin'] = bin
        return data['bin']

    def trans_to_woe(self, X):
        '''
        新值分到缺失
        X: series
        '''
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        if not is_string_dtype(X):
            X = X.astype(str)
        data = X.to_frame()
        woe = self.__bin_stats['WoE'].sort_index()
        data['woe'] = woe.iloc[0]
        for bin, values in enumerate(self.__bins):
            data.loc[X.isin(values), 'woe'] = woe[bin]
        return data['woe']

    def __cc(self, dfx):
        cnt = len(dfx)
        bad = dfx.YY.sum()
        good = cnt - bad
        return pd.Series(
            {'Var': self.__varname, 'Range': dfx.XX.unique(), 'CntRec': cnt, 'CntGood': good, 'CntBad': bad})

    def calc_stats(self, data):
        '''
        计算woe，iv等。
        '''
        res = data.groupby(data['bin']).apply(self.__cc)
        cntg = (data.YY == 0).sum()
        cntb = (data.YY == 1).sum()
        res['Pct'] = res.CntRec / len(data)
        res['PctBad'] = res.CntBad / cntb
        res['PctGood'] = res.CntGood / cntg
        res['BadRate'] = res.CntBad / res.CntRec
        res['CumGood'] = res.CntGood.cumsum()
        res['CumBad'] = res.CntBad.cumsum()
        res['Odds'] = res.BadRate / (1 - res.BadRate)
        res['LnOdds'] = np.log(res.Odds)
        res['WoE'] = np.log(res.PctBad / res.PctGood)
        res['IV'] = (res.PctBad - res.PctGood) * res.WoE
        res['TotalIV'] = res.IV.replace({np.inf: 0, -np.inf: 0}).sum()
        return res

    def manual_bin(self, df, x, y, bins=[]):
        '''
        手动分箱
        df: 数据
        x: 变量名
        y: 目标变量
        bins: [['a'],['b'],['c','d'],['e']]
        '''
        self.__varname = x
        data = pd.DataFrame({'XX': df[x], 'YY': df[y]})
        assert data.YY.isin([0, 1]).all(), 'ERROR: {} 目标变量非0/1!'.format(y)
        if not is_string_dtype(data.XX):
            data['XX'] = data.XX.astype(str)
        cnt = data.XX.count()
        assert cnt > 0, 'ERROR: "{}" 变量值全为 NULL  !'.format(x)
        data['bin'] = -1
        for i, values in enumerate(bins):
            data.loc[data.XX.isin(values), 'bin'] = i
        data.loc[data.XX.isnull(), 'bin'] = -2
        self.__bins = bins.copy()
        res = self.calc_stats(data)
        self.__bin_stats = res

    def pct_bin(self, df, x, y, sp_bins=[], max_bin=10):
        '''
        字符型自动分箱，
        sp_bins: 特殊值分箱. [['a'],['b'],['c','d'],['e']]
        df: 数据
        x: 变量名
        y: 目标变量
        '''
        spvars = []
        for binb in sp_bins:
            spvars = spvars + binb
        assert len(set(spvars)) == len(spvars), 'ERROR: "{}" : sp_bins are overlapping!'.format(x)
        data = pd.DataFrame({'XX': df[x], 'YY': df[y]})
        assert data.YY.isin([0, 1]).all(), 'ERROR: {} 目标变量非0/1!'.format(y)
        data = data.dropna()
        cnt = data.shape[0]
        assert cnt > 0, 'ERROR: "{}" 变量值全为 NULL  !'.format(x)
        if not is_string_dtype(data.XX):
            data['XX'] = data.XX.astype(str)
        nuniq = data.XX.nunique()
        if nuniq > 50:
            print('WARN: "{}" 字符型变量取值数超过 {} 个!'.format(x, nuniq))

        db = data[~data.XX.isin(spvars)]
        dbr = db.groupby('XX').YY.mean().reset_index()
        dbr['rr'] = dbr.YY.rank(pct=1)
        edges = pd.Series(np.linspace(0, 1, max_bin + 1))
        dbr['bin'] = dbr.rr.map(lambda r: (edges >= r).idxmax())
        xx = dbr.groupby('bin').apply(lambda yy: yy.XX.tolist())
        sp_bins = sp_bins + xx.tolist()
        self.manual_bin(df, x, y, sp_bins.copy())


def calc_chi2(arr):

    """
    计算卡方值

    arr:频数统计表,二维numpy数组。

    """

    assert(arr.ndim ==2)
    # 计算每行总频数
    R_N = arr.sum(axis=1)

    # 每列总频数
    C_N = arr.sum(axis=0)

    # 总频数
    N = arr.sum()

    # 计算期望频数 C_i * R_j / N。
    E = np.ones(arr.shape) * C_N / N

    E = (E.T * R_N).T

    square = (arr - E) ** 2 / E

    # 期望频数为0时，做除数没有意义，不计入卡方值
    square[E == 0] = 0

    # 卡方值
    v = square.sum()

    return v


def chiMerge(df, col, target, max_groups=None, threshold=None):
    """
    卡方分箱

    df: pandas dataframe数据集

    col: 需要分箱的变量名（数值型）

    target: 类标签

    max_groups: 最大分组数。

    threshold: 卡方阈值，如果未指定max_groups，默认使用置信度95%设置threshold。

    return: 包括各组的起始值的列表.

    """

    freq_tab = pd.crosstab(df[col], df[target])

    # 转成numpy数组用于计算。

    freq = freq_tab.values

    # 初始分组切分点，每个变量值都是切分点。每组中只包含一个变量值.
    # 分组区间是左闭右开的，如cutoffs = [1,2,3]，则表示区间 [1,2) , [2,3) ,[3,3+)。
    cutoffs = freq_tab.index.values

    # 如果没有指定最大分组
    if max_groups is None:

        # 如果没有指定卡方阈值，就以95%的置信度（自由度为类数目-1）设定阈值。
        if threshold is None:
            # 类数目

            cls_num = freq.shape[-1]

            threshold = chi2.isf(0.05, df=cls_num - 1)

    while True:

        minvalue = None

        minidx = None

        # 从第1组开始，依次取两组计算卡方值，并判断是否小于当前最小的卡方
        for i in range(len(freq) - 1):
            v = calc_chi2(freq[i:i + 2])

        if minvalue is None or minvalue > v:
            # 小于当前最小卡方，更新最小值

            minvalue = v

            minidx = i

        # 如果最小卡方值小于阈值，则合并最小卡方值的相邻两组，并继续循环
        if (max_groups is not None and max_groups < len(freq)) or (threshold is not None and minvalue < threshold):

            # minidx后一行合并到minidx

            tmp = freq[minidx] + freq[minidx + 1]

            freq[minidx] = tmp

            # 删除minidx后一行
            freq = np.delete(freq, minidx + 1, 0)

            # 删除对应的切分点
            cutoffs = np.delete(cutoffs, minidx + 1, 0)

        else:
            # 最小卡方值不小于阈值，停止合并。

            break

    return cutoffs


if __name__ == '__main__':

    e_data = ExampleData()

    df = e_data.get_iris2()

    nb = XXDNumberBin()

    # nb.pct_bin(df, 'sepal_width', 'y')

    print(nb.get_bin_stats())