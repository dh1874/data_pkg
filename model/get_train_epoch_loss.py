#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-06-03
# @Author  : HD
import sys
from io import StringIO

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from utils.example_data import ExampleData

if __name__ == '__main__':

	"""
	截取，模型训练的中间过程。
	"""

	edd = ExampleData()

	df = edd.get_barest_cancer_data()
	cols = df.columns
	old_stdout = sys.stdout
	sys.stdout = mystdout = StringIO()
	clf = GradientBoostingRegressor(verbose=1)
	clf.fit(df[cols[:-2]], df[cols[-1]])
	sys.stdout = old_stdout
	loss_history = mystdout.getvalue()

	print(type(loss_history))
	# for i in loss_history.split('\n'):
	# 	print(i, len(i))
	print([{'epoch':j[7:10].strip(' '), 'loss': j[21:27]}
	       for i, j in enumerate(loss_history.split('\n'))
	       if i > 0 and len(j) >0 ])
	# print(loss_history)