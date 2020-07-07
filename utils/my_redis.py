#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-06-16
# @Author  : HD

import pickle
from redis.sentinel import Sentinel


class Reorder(object):

	def __init__(self, v):
		self.a = 1 + v
		self.b = 2 + v
		self.c = 3 + v
		self.d = 4 + v
		self.e = 5 + v
		self.f = 6 + v
		self.g = 7 + v
		self.h = 8 + v
		self.i = 9 + v
		self.j = 10 + v
		self.k = 11 + v
		self.l = 12 + v
		self.m = 13 + v
		self.n = 14 + v
		self.o = 15 + v
		self.p = 16 + v
		self.q = 17 + v
		self.r = 18 + v
		self.s = 19 + v
		self.t = 20 + v


class Smt(object):

	last_v = 0
	last_v_list = [1,2,3]

	def __init__(self, v):

		self.r = Reorder(v)
		self.last_v = v
		self.last_v_list.append(v)


class MyRedis(object):

	def __init__(self):
		# 主数据库别名（根据自己爱好设置的，叫狗蛋也挺好）
		service_name = 'mymaster'
		#
		sentinel = ([('192.168.1.32', 19111),
					 ('192.168.1.32', 19112),
		             ('192.168.1.32', 19113)])

		sentinel = Sentinel([('192.168.1.32', 19111),
							 ('192.168.1.32', 19112),
				             ('192.168.1.32', 19113)],
		                     socket_timeout=0.1)

		# rc = Sentinel(sentinel)

		print(sentinel.discover_master(service_name))
		print(sentinel.discover_slaves(service_name))

		# # 通过哨兵获取redis主从
		self.redis_master = sentinel.master_for(service_name=service_name,password=123456)
		self.redis_slave = sentinel.slave_for(service_name=service_name,password=123456)

		# master = rc.discover_master(service_name)

		# print(rc.discover_slaves(service_name))

		# print(master)


if __name__ == '__main__':

	s1 = Smt(1)

	s2 = Smt(2)

	s1_pickle = pickle.dumps(s1)

	s2_pickle = pickle.dumps(s2)

	print(s1_pickle)

	print(s2_pickle)

	mr = MyRedis()

	mr.redis_master.set('aaaaaaa', 123)

	print(mr.redis_slave.get('aaaaaaa'))

	mr.redis_master.set('aaaaaaa', 321)

	print(mr.redis_slave.get('aaaaaaa'))

	# mr.redis_master.set('s1_p', s1_pickle)
	#
	# res = mr.redis_slave.get('s2_p')
	#
	# print(res is None)
	#
	# s11 = pickle.loads(mr.redis_slave.get('s1_p', password=123456))
	#
	# print(s11.last_v_list)
	#
	# # mr.redis_master.

