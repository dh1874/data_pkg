#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-16 09:35:46
# @Author  : HD

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine


class MySqlUtils(object):
    """数据库操作"""

    def __init__(self, user, password, host, port, db):
        super(MySqlUtils, self).__init__()

        self.engine_str = 'mysql+pymysql://%s:%s@%s:%d/%s?charset=utf8' \
                          % (user, password, host, port, db)

        self.conn: sqlalchemy.engine.base.Engine = create_engine(self.engine_str,
                                                                 encoding='utf-8')

    def check_ping(self):
        # pool.Pool().recreate()
        pass

    def read_mysql_table(self, input_sql):
        """
        读取表
        :return pd.DataFrame
        """
        return pd.read_sql(input_sql, con=self.conn).fillna(0.0)

    def execute_sql(self, input_sql):
        """
        删除/更新语句
        :param input_sql: 输入的sql
        """

        self.conn.execute(input_sql)

    def write_data_into_mysql(self, df, tbl_name):
        """
        写入数据库, 追加
        """
        pd.DataFrame.to_sql(df, name=tbl_name, con=self.conn,
                            if_exists='append', index=False)


if __name__ == '__main__':

    pass