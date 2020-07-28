#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-27
# @Author  : HD
# uwsgi --http :9090 --wsgi-file foobar.py


def application(env, start_response):
    start_response('200 OK', [('Content-Type','text/html')])
    return [b"Hello World"]