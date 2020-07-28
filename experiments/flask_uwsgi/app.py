#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-27
# @Author  : HD
from flask import Flask
import socket
from flask import Blueprint
import time

app = Flask(__name__)


route_index = Blueprint('index_for', __name__)


@route_index.route('/for', methods=['POST', 'GET'])
def for_example():
	for i in range(20):
		time.sleep(1)
		print(i)
	return 'lalala'


app.register_blueprint(route_index)


def get_host_ip():
	try:
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		s.connect(('8.8.8.8', 80))
		ip = s.getsockname()[0]
	finally:
		s.close()
	return ip


@app.route('/hello', methods=['POST', 'GET'])
def hello():

	print('hello world!')

	return 'lalala'


@app.route('/anew', methods=['POST'])
def new():
	print('newnnewnwenewnwen')
	return 'new'


if __name__ == '__main__':

	ip = get_host_ip()

	app.run()