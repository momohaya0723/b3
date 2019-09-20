from flask import Blueprint, jsonify, request
from random import *

api = Blueprint('api', __name__)

@api.route('/hello/<string:name>/')
def say_hello(name):
    response = { 'msg': "Hello {}".format(name) }
    return jsonify(response)

@api.route('/random')
def random_number():
    response = {
        'randomNumber': randint(1, 100)
    }
    return jsonify(response)

@api.route('/image')
def image_get():

    from face_detect import take, detect, prepro, feature, realtime
    take()
    detect()
    prepro()
    lbp = feature()
    response = {
        'randomNumber': randint(1, 100),
        'feature': lbp,
    }
    return jsonify(response)