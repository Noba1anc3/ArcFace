# -*- coding: utf-8 -*-
# Copyright(c) 2018-present, Videt Tech. All rights reserved.
# @Project : EDU_PRODUCT
# @Time    : 19-4-10 上午10:10
# @Author  : kongshuchen
# @FileName: face_Solver.py
# @Software: PyCharm
from flask import Flask, current_app
from flask import request
import cv2
import base64
from face_Solver import Solver
from flask_cors import CORS
import json
import config
import numpy as np
# from config import save_path
import time
import os

# from gevent import pywsgi
# from gevent import monkey
# monkey.patch_all()

app = Flask(__name__)
CORS(app)
handle = None


@app.route('/')
def index():
    return "Hello, World!"


@app.route('/face_detect', methods=['POST'])
def face_detect():
    img = request.data
    image = cv2.imdecode(np.fromstring(base64.b64decode(img), dtype=np.uint8), 1)
    res = handle.response(image)
    res = json.dumps(res)
    return res



if __name__ == '__main__':
    # init model
    handle = Solver(config)

    # server = pywsgi.WSGIServer(('0.0.0.0',config.PORT),app)
    # server.serve_forever()
    app.run(host=config.HOST, port=config.PORT, debug=False,
            threaded=True)  # , ssl_context=("./service.videt.cn/certificate.crt","./service.videt.cn/private.key"))
