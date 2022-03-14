import os
from flask import Flask, request, jsonify, make_response
from threading import Lock
from model import train, evaluate, predict

app = Flask(__name__)
app.secret_key = os.urandom(128)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
LOCK = Lock()


@app.route('/api/predict', methods=['GET'])
def process():
    user_id = request.args.get('user_id', default=100, type=int)

    # if LOCK.locked():
    #     return make_response(jsonify({'error': 'Processing in progress!'}), 403)
    #
    # with LOCK:
    #     try:
    #         result = predict(user_id)
    #     except Exception as e:
    #         print(e)
    #         return make_response(jsonify({'error': f'{e}'}), 500)
    #     else:
    #         return make_response(jsonify({'result': f'/{user_id}/' + ' '.join(result)}))


@app.route('/api/log', methods=['GET'])
def get_logs_tail():
    pass

@app.route('api/info', methods=['GET'])
def get_service_info():
    pass

@app.route('api/reloac', methods=['GET'])
def reload_model():
    pass


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
