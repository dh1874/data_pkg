#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-25
# @Author  : HD


import pandas as pd
from sklearn.model_selection import train_test_split


class TrainTestSplit(object):
	"""
	构造训练测试集plot.py
	"""

	def __init__(self, trainData, yname, is_test_size=None):
		super(TrainTestSplit, self).__init__()
		self.yName = yname
		self.copy_odds = 0  # 訓練集中y=1樣本是否自我複製
		self.under_odds = 0  # 欠抽樣比例
		self.smote_odds = 0
		self.train = trainData
		self.train_1 = trainData[trainData[yname] == 1]
		self.train_0 = trainData[trainData[yname] == 0]
		self.is_test_size = is_test_size  # 是否要劃分出測試集的比例大小
		self.nums_1 = len(self.train_1)
		self.nums_0 = len(self.train_0)
		self.test = None

	def set_overSample_smoteOdds(self):
		'''smote包的过抽样'''

	def set_overSample_copyOdds(self, input):
		'''y=1樣本複製次數'''
		self.copy_odds = input
		return self

	def set_underSample_odds(self, input):
		'''y=0样本之于y=1样本的倍数'''
		self.under_odds = input
		return self

	# 按配置的實際數值，執行训练测试集生成的預處理
	@property
	def trainTestPreProcess_(self):
		'''訓練測試集的抽樣預處理

		return：
		——————————
		self.trian
		self.test

		'''
		print("[HD_Pre] start data preProcess")
		print("[HD_Pre] start TrainData nums_1 : nums_0 = %s : %s " % (self.nums_1, self.nums_0))

		# 輸入test==null, 則將train隨機拆分
		if self.is_test_size:
			print('[HD_Pre] train test random split')

			train_size = 1 - self.is_test_size

			trianTest = train_test_split(self.train,
										 train_size=train_size, test_size=self.is_test_size)

			self.train = trianTest[0]

			self.test = trianTest[1]

			self.train_1 = self.train[self.train[self.yName] == 1]

			self.train_0 = self.train[self.train[self.yName] == 0]

			self.nums_1 = len(self.train_1)

			self.nums_0 = len(self.train_0)

		# y=1樣本複製[使用粗暴的自我复制， 有空的时候把它改成SMOTE算法]
		if self.copy_odds > 0:

			print("[HD_Pre] start copy sample of y=1")

			for i in range(0, self.copy_odds, 1):
				print("[HD_Pre] copy %s" % (i + 1))

				self.train_1 = pd.concat([self.train_1, self.train_1])

			self.nums_1 = len(self.train_1)

		# 訓練集前抽樣
		if self.under_odds > 0:
			print("[HD_Pre] start underSample")

			samplePct = self.nums_1 * self.under_odds / self.nums_0

			samplePct = samplePct if samplePct < 1 else 1

			self.train_0 = self.train_0.sample(frac=samplePct)

			self.train = pd.concat([self.train_1, self.train_0])

			self.nums_0 = self.under_odds * self.nums_1

		print("[HD_Pre] end TrainData nums_1 : nums_0 = %s : %s" % (self.nums_1, self.nums_0))

		return self

	@property
	def getTrain_(self):

		return self.train

	@property
	def getTest_(self):

		return self.test

