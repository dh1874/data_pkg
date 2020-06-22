from math import *
from decimal import Decimal


"""
相似度
"""


def euclidean_distance(x, y):
    """
    欧式距离

    :param x:
    :param y:
    :return:
    """
    return sqrt(sum(pow(a-b, 2) for a, b in zip(x, y)))
 

def square_rooted(x):
    """
    向量平方根
    :param x:
    :return:
    """
    return round(sqrt(sum([a*a for a in x])),3)
 

def cosine_similarity(x, y):
    """
    余弦相似度

    :param x:
    :param y:
    :return:
    """
    numerator = sum(a*b for a, b in zip(x, y))

    denominator = square_rooted(x) * square_rooted(y)

    return round(numerator/float(denominator),3)


def jaccard_similarity(x, y):
    """
    杰卡德相似

    :param x:
    :param y:
    :return:
    """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))

    union_cardinality = len(set.union(*[set(x), set(y)]))

    return intersection_cardinality/float(union_cardinality)


def nth_root(value, n_root):
    """
    两个向量之间的闵氏距离

    :param value:
    :param n_root:
    :return:
    """
    root_value = 1/float(n_root)

    return round(Decimal(value) ** Decimal(root_value), 3)


def minkowski_distance(x, y, p_value):
    return nth_root(sum(pow(abs(a-b), p_value) for a, b in zip(x, y)), p_value)


if __name__ == '__main__':
    x = [0, 3, 4, 5]

    y = [7, 6, 3, -1]

    p_value = 3

    print("欧式距离 ： %s" % euclidean_distance(x, y))

    print("余弦相似度 ： %s" % cosine_similarity(x, y))

    print("杰卡德相似度 ： %s" % jaccard_similarity(x, y))

    print("闵氏距离 ： %s" % minkowski_distance(x, y, p_value))
