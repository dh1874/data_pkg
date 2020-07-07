#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-06-10
# @Author  : HD


def find_the_key_of_min_value_in_dict(the_dict):
    """
    找出字典中，value最小的key值

    :param the_dict: {'a': 3, 'b': 2, 'c': 1}
    :return: 'c'
    """

    return min(the_dict, key=the_dict.get)


def division():

    try:

        return (None +1) /2

    except:

        return 0.0


if __name__ == '__main__':

    print(division())