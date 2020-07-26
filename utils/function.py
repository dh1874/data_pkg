#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-06-10
# @Author  : HD
import collections


def find_the_key_of_min_value_in_dict(the_dict):
    """
    找出字典中，value最小的key值

    :param the_dict: {'a': 3, 'b': 2, 'c': 1}
    :return: 'c'
    """

    return min(the_dict, key=the_dict.get)

def count_list_1(alist):
    """
    计数
    
    :param alist: list
    :return dict{v: cnt}
    """

    return collections.Counter(alist)

def count_list_1(alist):
    """
    计数
    
    :param alist: list
    :return dict{v: cnt}
    """

    res_dict = {}

    for i in alist:

        res_dict = res_dict.get(i, 0) += 1


def sorted_dict_by_value(the_dict):

    return sorted(dicts.items(), key=lambda item:item[1], reverse=True)




def division():

    try:

        return (None +1) /2

    except:

        return 0.0


if __name__ == '__main__':

    dicts = count_list([1,1,1,3,3,3,3,2])
    # sorted(dic.items(), key=lambda item:item[1], reverse=True)
    print(sorted_dict_by_value(dicts))