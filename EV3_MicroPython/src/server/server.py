from flask import Flask, jsonify
import json
import os

app = Flask(__name__)

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

@app.route('/balls', methods=['GET'])
def get_balls():
    try:
        data = load_data('../data/balls_positions/balls_positions.json')
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/robots', methods=['GET'])
def get_robots():
    try:
        data = load_data('../data/robots_positions/robots_positions.json')
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/field', methods=['GET'])
def get_field():
    try:
        data = load_data('../data/field_positions/field_positions.json')
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/obstacles', methods=['GET'])
def get_obstacles():
    try:
        data = load_data('../data/obstacle_positions/obstacle_positions.json')
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
