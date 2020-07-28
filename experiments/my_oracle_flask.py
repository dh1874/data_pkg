#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-03
# @Author  : HD
# from utils.mysql_util import MySqlUtils
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Sequence, create_engine
import pandas as pd
from sqlalchemy import Column, Integer, String

engine_str = 'oracle://%s:%s@%s:%d/?service_name=%s' % ('pca', '1234', '192.168.1.29', 1521, 'xe')
print(engine_str)
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = engine_str
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class ModelDetailNew(db.Model):
	__tablename__ = 'model_detail_copy'
	id_seq = Sequence('id_seq')
	id = db.Column(db.Integer, id_seq, primary_key=True)
	detail = db.Column(db.Integer)


#db.create_all()

m = ModelDetailNew()

print('-------',m)
m.id = 1

m.detail = 111

print(m)
print(m.detail, type(m.detail))


db.session.add(m)
db.session.commit()


if __name__ == '__main__':

	#app.run()
	user = 'pca'
	password = '1234'
	host = '192.168.1.29'
	port = '1521'
	db = 'xe'

	engine_str = 'oracle+cx_oracle://%s:%s@%s:%d/?service_name=%s' % (user, password, host, port, db)

	conn = create_engine(engine_str, encoding='utf-8')

	df = pd.read_sql('select * from model_detail_copy', conn)

	print(df)

	df['id'] = 666

	df['detail'] = 666

	df.to_sql('model_detail', con=conn, if_exists='append', index=False)

