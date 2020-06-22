#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-04-17
# @Author  : HD

import logging
import os
from utils.configuration import *


class MyLogger(object):

    # 文件路径
    file_path = ''

    # 文件名称
    file_name = '2222.txt'

    # 日志级别关系映射
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }


    """
    format：指定日志信息的输出格式，即上文示例所示的参数，详细参数可以参考：docs.python.org/3/library/l…，部分参数如下所示：
        %(levelno)s：打印日志级别的数值。
        %(levelname)s：打印日志级别的名称。
        %(pathname)s：打印当前执行程序的路径，其实就是sys.argv[0]。
        %(filename)s：打印当前执行程序名。
        %(funcName)s：打印日志的当前函数。
        %(lineno)d：打印日志的当前行号。
        %(asctime)s：打印日志的时间。
        %(thread)d：打印线程ID。
        %(threadName)s：打印线程名称。
        %(process)d：打印进程ID。
        %(processName)s：打印线程名称。
        %(module)s：打印模块名称。
        %(message)s：打印日志信息。
    """

    # '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    FMT = '[%(filename)s line:%(lineno)d] - %(levelname)s: %(message)s'

    def __init__(self):
        super(object, self).__init__()

    def config(self):
        pass

    def init_path(self):
        # 目录是否存在, 不存在则创建
        mkdir_lambda = lambda x: os.makedirs(x) if not os.path.exists(x) else True
        # self.file_name = self.plan_name + '.log'
        # # 根目录/CATEGORY/yyyyMMdd/USER_ID/PLAN_NAME.log
        self.file_path = LOG_FILE_PHAT + '/' + self.file_name
        # 創建
        mkdir_lambda(self.file_path)

    def init_logger(self):

        # logging.basicConfig(level=logging.INFO, format=self.FMT)

        logger = logging.getLogger()

        # StreamHandler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level=logging.INFO)
        logger.addHandler(stream_handler)

        # FileHandler
        self.init_logger()
        file_handler = logging.FileHandler(self.file_path + '/' + self.file_name)
        file_handler.setLevel(level=logging.INFO)
        logger.addHandler(file_handler)

        return logger


if __name__ == '__main__':  

    # mlog = MyLogger()
    #
    # logger = mlog.init_logger()
    #
    # logger.info('asdf')