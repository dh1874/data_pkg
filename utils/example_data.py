#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-13 15:30:45
# @Author  : HD 
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import datasets


class ExampleData(object):
    """
    DEMO数据
    """

    @staticmethod
    def get_iris_data():
        """
        [sepal_length  sepal_width  petal_length  petal_width species]
        :return:
        """

        return sns.load_dataset('iris')

    @staticmethod
    def get_iris2():

        df_iris = sns.load_dataset('iris')

        df_iris['y'] = list(map(lambda x: 1 if x == 'virginica' else 0, df_iris['species']))

        df_iris['id'] = 1

        return df_iris

    @staticmethod
    def get_barest_cancer_data():
        """
        np.c_ is the numpy concatenate function
        :return: dataframe
        """

        data = datasets.load_breast_cancer()

        return pd.DataFrame(data=np.c_[data['data'], data['target']],
                            columns=list(data['feature_names']) + ['target'])

    @staticmethod
    def get_boston():
        """
        :return: dataframe
        """

        data = datasets.load_boston()

        return pd.DataFrame(data=np.c_[data['data'], data['target']],
                            columns=list(data['feature_names']) + ['target'])


if __name__ == '__main__':

    e_data = ExampleData()

    df = e_data.get_barest_cancer_data()

    iris = e_data.get_iris_data()

    print(iris[['species']].drop_duplicates())

    # print(df.head())
    #
    # count = pd.crosstab(df['mean radius'], df['target'])
    #
    # print(count)
    #
    # print(count.sum(axis=1))
    #
    # print(count[1])
