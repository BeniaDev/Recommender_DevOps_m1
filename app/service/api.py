import os
from flask import Flask, request, jsonify, make_response
from threading import Lock
from model import predict, reload
import json
import logging

logging.basicConfig(filename='../logs/app.log', level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

app = Flask(__name__)
LOCK = Lock()


@app.route('/api/log')
def get_logs_tail():
    with open("../logs/app.log", "r") as log_f:
        logs_tail = log_f.readlines()[-20:]

    if LOCK.locked():
        return make_response(jsonify({'error': 'Processing in progress!'}), 403)

    with LOCK:
        return make_response(jsonify({'Last 20s app.log rows': logs_tail}))


@app.route('/api/predict', methods=['GET'])
def process():
    user_id = request.args.get('user_id', default=100, type=int)
    M_items_recommend = request.args.get('M', default=20, type=int)

    if LOCK.locked():
        return make_response(jsonify({'error': 'Processing in progress!'}), 403)

    with LOCK:
        try:
            movies, ratings = predict(user_id, M=M_items_recommend)
        except Exception as e:
            print(e)
            return make_response(jsonify({'error': f'{e}'}), 500)
        else:
            return make_response(jsonify({'result': f"Movies: {movies}  With Ratings: {ratings}"}))


@app.route('/api/info', methods=['GET'])
def get_service_info():
    if LOCK.locked():
        return make_response(jsonify({'error': 'Processing in progress!'}), 403)

    with LOCK:
        return make_response(jsonify({'Service_info': "Will be here soon :-)"}))

@app.route('/api/reload', methods=['GET'])
def reload_model():
    if LOCK.locked():
        return make_response(jsonify({'error': 'Processing in progress!'}), 403)

    with LOCK:
        reload()
        return make_response(jsonify({'Result': "Model successfully reloaded!"}))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
