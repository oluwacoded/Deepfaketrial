"""
DeepFaceLive Web Server
Flask + SocketIO with eventlet (required for WebSocket support).
"""
import eventlet
eventlet.monkey_patch()  # must be first, before any other imports

import base64
import os

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

from web_pipeline import FaceSwapPipeline

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'deepfacelive-web')
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB upload cap
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet',
                    max_http_buffer_size=10 * 1024 * 1024)

ALLOWED_IMAGE_MIMES = {'image/jpeg', 'image/png', 'image/webp', 'image/gif'}

pipeline = FaceSwapPipeline()

# -----------------------------------------------------------------------
# REST endpoints
# -----------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/models')
def api_models():
    return jsonify(pipeline.get_available_models())


@app.route('/api/model/status')
def api_model_status():
    return jsonify(pipeline.get_model_status())


@app.route('/api/model/load', methods=['POST'])
def api_model_load():
    data = request.get_json(force=True)
    name = data.get('name', '')
    if not name:
        return jsonify({'error': 'name required'}), 400
    pipeline.load_model(name)
    return jsonify({'ok': True})


@app.route('/api/model/unload', methods=['POST'])
def api_model_unload():
    pipeline.unload_model()
    return jsonify({'ok': True})


@app.route('/api/settings', methods=['POST'])
def api_settings():
    data = request.get_json(force=True)
    if 'morph_factor' in data:
        pipeline.set_morph_factor(float(data['morph_factor']))
    if 'face_opacity' in data:
        pipeline.set_face_opacity(float(data['face_opacity']))
    if 'color_transfer' in data:
        pipeline.set_color_transfer(data['color_transfer'])
    if 'enabled' in data:
        pipeline.set_enabled(bool(data['enabled']))
    return jsonify({'ok': True})


@app.errorhandler(413)
def too_large(_):
    return jsonify({'error': 'Image too large (max 20 MB)'}), 413


@app.route('/api/process_image', methods=['POST'])
def api_process_image():
    """Accept an uploaded image, run face swap, return result as base64 JPEG."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    mime = file.content_type or ''
    if mime and mime not in ALLOWED_IMAGE_MIMES:
        return jsonify({'error': f'Unsupported file type: {mime}'}), 415

    jpg_bytes = file.read()
    if not jpg_bytes:
        return jsonify({'error': 'Empty file'}), 400

    result_bytes, face_found = pipeline.process_frame(jpg_bytes)
    result_b64 = 'data:image/jpeg;base64,' + base64.b64encode(result_bytes).decode()
    return jsonify({'image': result_b64, 'face_found': face_found})


# -----------------------------------------------------------------------
# WebSocket – frame processing
# -----------------------------------------------------------------------

@socketio.on('frame')
def handle_frame(data):
    """
    Client sends: { 'image': '<base64 JPEG>' }
    Server emits back: { 'image': '<base64 JPEG>', 'face_found': bool }
    """
    try:
        b64 = data.get('image', '')
        if not b64:
            return
        if ',' in b64:
            b64 = b64.split(',', 1)[1]

        jpg_bytes = base64.b64decode(b64)
        result_bytes, face_found = pipeline.process_frame(jpg_bytes)
        result_b64 = 'data:image/jpeg;base64,' + base64.b64encode(result_bytes).decode()
        emit('frame_result', {'image': result_b64, 'face_found': face_found})
    except Exception as e:
        print(f'[WS] frame error: {e}')
        emit('frame_error', {'message': str(e)})


@socketio.on('connect')
def on_connect():
    print(f'[WS] Client connected: {request.sid}')


@socketio.on('disconnect')
def on_disconnect():
    print(f'[WS] Client disconnected: {request.sid}')


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == '__main__':
    print('Starting DeepFaceLive Web on http://0.0.0.0:5000')
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
