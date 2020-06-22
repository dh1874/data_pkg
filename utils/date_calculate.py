#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-25
# @Author  : HD

import calendar
import datetime
import pandas as pd


def get_today_str(date_format='%Y-%m-%d'):
    """
    :param date_format: '%Y-%m-%d'
    :return: str 'yyyy-MM-dd'
    """

    return datetime.datetime.now().strftime(date_format)


class DateUtil(object):
    """日期处理"""

    DATE_FORMAT = '%Y-%m-%d'
    date_str = 0
    date_time = 0
    year = 0
    month = 0
    weekday = 0
    days = 0

    def __init__(self):
        super(DateUtil, self).__init__()

    def set_date_str(self, date_str, DATE_FORMAT=None):
        """
        加载要被处理的日期

        :param date_str: ‘yyyy-mm-dd’
        :return:
        """
        if DATE_FORMAT:
            self.DATE_FORMAT = DATE_FORMAT
        self.date_str = date_str
        self.date_time = self.get_datetime(date_str)
        self.year = self.date_time.year
        self.month = self.date_time.month
        self.weekday = self.date_time.weekday() + 1
        self.days = self.cal_days_of_month(self.year, self.month)

    def get_gap_date_str(self, days_gap):
        """获取多少天前，多少天后"""
        dt = self.date_time + datetime.timedelta(days=days_gap)

        return dt.strftime(self.DATE_FORMAT)

    def cal_week_nbr_of_month(self, year, mth, end_day):
        """计算某年月的第一天到end_day之间共多少周"""

        # 总周数
        total_weeks_number = 0

        """
        遍历1 - 月末， 共计多少个周三， 则有多少周
        """
        for day_i in range(1, end_day + 1):

            dt_i = self.get_datetime('%s-%s-%s' % (year, mth, day_i))

            if dt_i.weekday() + 1 == 3:

                total_weeks_number += 1

        return total_weeks_number

    @property
    def remain_days_of_current_week_(self):
        """输入date_str的，当周剩余天数"""
        return 7 - self.weekday

    @property
    def workdays_to_today_(self):
        """输入date_str的，月初1号 至今 多少个工作日"""

        begin_str = self.date_time.replace(day=1).strftime(self.DATE_FORMAT)

        return self.cal_workdays_in_date_range(begin_str, self.date_str) - 1

    @property
    def total_weeks_of_current_mth_(self):
        """输入date_str的，该天所在月共多少周"""
        year = self.date_time.year
        mth = self.date_time.month

        return self.cal_week_nbr_of_month(year, mth, self.cal_days_of_month(year, mth))

    @property
    def the_week_of_current_mth_(self):
        """输入date_str的，该天在该月第几周"""
        year = self.date_time.year
        mth = self.date_time.month

        return self.cal_week_nbr_of_month(year, mth, self.date_time.day)

    @property
    def total_weeks_next_mth_(self):
        """输入date_str的，下月总周数"""
        next_mth = self.month + 1 if self.month < 12 else 1
        years = self.year if self.month < 12 else self.year + 1
        days = self.cal_days_of_month(years, next_mth)
        return self.cal_week_nbr_of_month(years, next_mth, days)

    @property
    def total_weeks_last_mth_(self):
        """
        输入date_str的，上月总周数
        :return:
        """
        next_mth = self.month - 1 if self.month > 1 else 12
        years = self.year if self.month > 1 else self.year - 1
        days = self.cal_days_of_month(years, next_mth)
        return self.cal_week_nbr_of_month(years, next_mth, days)

    @property
    def last_month_sales_days_(self):
        """
        输入date_str的，输入日期的上月销售天数
        :return:
        """
        last_month = self.month - 1
        if self.month == 1:
            last_month = 12
            self.year = self.year - 1

        # 该天所在月共计天数
        number_days = self.cal_days_of_month(self.year, last_month)

        end_str = '%s-%s-%s' % (self.year, last_month, number_days)

        dt_i = self.get_datetime(end_str)

        start_str = dt_i.replace(day=1).strftime('%Y-%m-%d')

        return self.cal_workdays_in_date_range(start_str, end_str)

    def get_datetime(self, date_str):
        """
        STR -> DATETIME
        :param date_str: yyyy-mm-dd
        :return:
        """
        return datetime.datetime.strptime(date_str, self.DATE_FORMAT)

    @staticmethod
    def cal_days_of_month(year, mth):
        """某年某月多少天"""

        month_range = calendar.monthrange(year, mth)

        return month_range[1]

    @staticmethod
    def cal_workdays_in_date_range(begin_str, end_str):
        """
        [begin, end] 闭区间内有多少个工作日

        :param begin_str: date_str
        :param end_str: date_str
        :return:
        """

        return len(pd.bdate_range(begin_str, end_str))

    def days_current_month(self):
        # 该天所在月共计天数
        return self.days

    def today_month_day_reverse(self):

        return self.days_current_month() - int(self.date_time.strftime("%d"))


if __name__ == '__main__':

    date_util = DateUtil()

    date_util.set_date_str('2020-05-25')

    print(date_util.year)
    print(date_util.month)
    print(date_util.days)
    print(date_util.weekday)

    print(date_util.total_weeks_of_current_mth_)
    print(date_util.the_week_of_current_mth_)
    print(date_util.total_weeks_last_mth_)
    print(date_util.total_weeks_next_mth_)

    print(date_util.get_gap_date_str(10))




