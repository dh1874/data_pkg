#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-06-08
# @Author  : HD
from faker import Faker

"""
制造假数据
"""


if __name__ == '__main__':

    f = Faker()

    print(f.date())