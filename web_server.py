"""
DeepFaceLive Web Server
Flask + SocketIO with eventlet (required for WebSocket support).

Key design point (fixes the hangs):
  Face-swap inference is CPU-heavy and would block eventlet's single event
  loop — freezing every other client AND the WebRTC call signaling while one
  frame is processed. We therefore run inference in a native worker thread via
  eventlet.tpool.execute(); the event loop stays responsive the whole time.
"""
import eventlet
eventlet.monkey_patch()  # must be first, before any other imports
from eventlet import tpool  # native worker-thread pool (import submodule explicitly)

import base64
import json
import os
import re
import tempfile
import threading
import time
import uuid

import cv2
import numpy as np

from flask import Flask, jsonify, render_template, request, send_file
from flask_socketio import SocketIO, emit, join_room
from werkzeug.utils import secure_filename

from web_pipeline import FaceSwapPipeline, MODELS_DIR

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'deepfacelive-web')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB upload cap (DFM models can be large)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet',
                    max_http_buffer_size=10 * 1024 * 1024)

ALLOWED_IMAGE_MIMES = {'image/jpeg', 'image/png', 'image/webp', 'image/gif'}
ALLOWED_VIDEO_EXTS = {'.mp4', '.webm', '.avi', '.mov', '.mkv', '.m4v'}

pipeline = FaceSwapPipeline()


def run_swap(jpg_bytes):
    """Snapshot model state in this (green) thread, then do the heavy CPU work
    in a native worker thread so the event loop never blocks."""
    snap = pipeline.snapshot()
    return tpool.execute(pipeline.run, jpg_bytes, snap)


# -----------------------------------------------------------------------
# Video-processing job registry (in-memory, single-process)
# -----------------------------------------------------------------------
_VIDEO_DIR = os.path.join(tempfile.gettempdir(), 'dfl_video_jobs')
os.makedirs(_VIDEO_DIR, exist_ok=True)
_video_jobs = {}
_video_jobs_lock = threading.Lock()


def _set_job(job_id, **fields):
    with _video_jobs_lock:
        job = _video_jobs.setdefault(job_id, {})
        job.update(fields)


def _get_job(job_id):
    with _video_jobs_lock:
        job = _video_jobs.get(job_id)
        return dict(job) if job is not None else None

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
    if 'face_output_size' in data:
        pipeline.set_face_output_size(int(data['face_output_size']))
    if 'enabled' in data:
        pipeline.set_enabled(bool(data['enabled']))
    return jsonify({'ok': True})


@app.errorhandler(413)
def too_large(_):
    return jsonify({'error': 'File too large (max 1 GB)'}), 413


@app.route('/api/model/upload', methods=['POST'])
def api_model_upload():
    """Accept an uploaded .dfm model file, save it to userdata/models/ so it
    appears in the model catalogue alongside the built-in ones."""
    if 'model' not in request.files:
        return jsonify({'error': 'No model file uploaded'}), 400

    file = request.files['model']
    filename = file.filename or ''
    if os.path.splitext(filename)[1].lower() != '.dfm':
        return jsonify({'error': 'Only .dfm files are supported'}), 415

    safe = secure_filename(filename)
    if not safe or not safe.lower().endswith('.dfm'):
        return jsonify({'error': 'Invalid file name'}), 400

    dest = os.path.join(str(MODELS_DIR), safe)
    file.save(dest)
    if os.path.getsize(dest) == 0:
        _safe_remove(dest)
        return jsonify({'error': 'Empty file'}), 400

    name = os.path.splitext(safe)[0]
    print(f'[Model] Uploaded custom model "{name}" ({os.path.getsize(dest)} bytes)')
    return jsonify({'ok': True, 'name': name})


def _read_upload():
    """Validate + read an uploaded 'image' file. Returns (bytes, error_json, status)."""
    if 'image' not in request.files:
        return None, {'error': 'No image uploaded'}, 400
    file = request.files['image']
    mime = file.content_type or ''
    if mime and mime not in ALLOWED_IMAGE_MIMES:
        return None, {'error': f'Unsupported file type: {mime}'}, 415
    jpg_bytes = file.read()
    if not jpg_bytes:
        return None, {'error': 'Empty file'}, 400
    return jpg_bytes, None, 200


@app.route('/api/process_image', methods=['POST'])
def api_process_image():
    """Accept an uploaded image, run face swap, return result as base64 JPEG."""
    jpg_bytes, err, status = _read_upload()
    if err:
        return jsonify(err), status
    result_bytes, face_found, mode = run_swap(jpg_bytes)
    result_b64 = 'data:image/jpeg;base64,' + base64.b64encode(result_bytes).decode()
    return jsonify({'image': result_b64, 'face_found': face_found, 'mode': mode})


@app.route('/api/target_face', methods=['POST', 'DELETE'])
def api_target_face():
    """POST: set a custom target face (paste mode). DELETE: clear it."""
    if request.method == 'DELETE':
        pipeline.clear_target_face()
        return jsonify({'ok': True})

    jpg_bytes, err, status = _read_upload()
    if err:
        return jsonify(err), status
    # detection is light but still offload to keep the loop responsive
    result = tpool.execute(pipeline.set_target_face, jpg_bytes)
    return jsonify(result)


# -----------------------------------------------------------------------
# Video call — WebRTC signaling relay
# -----------------------------------------------------------------------
# The caller (main page) publishes an SDP offer for a random room id and gets a
# shareable /call/<room_id> link. The callee opens that link, fetches the offer,
# and answers. We relay answer + ICE candidates between the two peers. ICE from
# the caller is buffered until the callee joins so no early candidates are lost.
#
# The store is bounded on every axis (room count, id format, payload size and
# buffered-candidate count) and rooms expire on TTL, on caller disconnect, on
# explicit End Call, and opportunistically — so signaling can't exhaust memory.

_rooms = {}                 # room_id -> {offer, ts, caller_sid, caller_ice, answered}
_ROOM_TTL = 1800            # seconds a room lingers unused
_MAX_ROOMS = 100            # hard cap on concurrent rooms (≈ concurrent callers)
_MAX_SDP_BYTES = 30_000     # a WebRTC offer/answer SDP is a few KB in practice
_MAX_ICE_BYTES = 2_000      # a single ICE candidate is tiny
_MAX_ICE_PER_ROOM = 30      # cap buffered caller candidates
_ROOM_ID_RE = re.compile(r'^[A-Za-z0-9_-]{4,64}$')
# Worst-case retained state: 100 rooms x (30KB offer + 30 x 2KB ICE) ≈ 9 MB.


def _valid_room_id(rid):
    return isinstance(rid, str) and bool(_ROOM_ID_RE.match(rid))


def _payload_too_big(obj, limit=_MAX_SDP_BYTES):
    try:
        return len(json.dumps(obj)) > limit
    except (TypeError, ValueError):
        return True


def _cleanup_rooms():
    now = time.time()
    for rid in [r for r, v in _rooms.items() if now - v['ts'] > _ROOM_TTL]:
        _rooms.pop(rid, None)


def _drop_rooms_for_sid(sid, reason='caller left'):
    for rid in [r for r, v in _rooms.items() if v['caller_sid'] == sid]:
        _rooms.pop(rid, None)
        print(f'[Call] Room {rid} dropped ({reason})')


@app.route('/call/<room_id>')
def call_page(room_id):
    return render_template('call.html', room_id=room_id)


@app.route('/api/room/<room_id>')
def api_room(room_id):
    _cleanup_rooms()
    room = _rooms.get(room_id)
    if not room:
        return jsonify({'error': 'Room not found or expired'}), 404
    return jsonify({'offer': room['offer']})


@socketio.on('webrtc_offer')
def on_webrtc_offer(data):
    data = data or {}
    room_id = data.get('room_id')
    offer = data.get('offer')
    if not _valid_room_id(room_id) or not offer or _payload_too_big(offer):
        return
    _cleanup_rooms()
    if room_id in _rooms and _rooms[room_id]['caller_sid'] != request.sid:
        emit('call_error', {'message': 'That room id is taken — please retry.'})
        return
    _drop_rooms_for_sid(request.sid, 'renewed')   # enforce one active room per caller
    if len(_rooms) >= _MAX_ROOMS:
        emit('call_error', {'message': 'Server is busy — please try again shortly.'})
        return
    _rooms[room_id] = {'offer': offer, 'ts': time.time(),
                       'caller_sid': request.sid, 'caller_ice': [], 'answered': False}
    join_room(room_id)
    emit('offer_stored', {'room_id': room_id})
    print(f'[Call] Offer stored for room {room_id} ({len(_rooms)} active)')


@socketio.on('webrtc_answer')
def on_webrtc_answer(data):
    data = data or {}
    room_id = data.get('room_id')
    answer = data.get('answer')
    if not _valid_room_id(room_id) or not answer or _payload_too_big(answer):
        return
    room = _rooms.get(room_id)
    if not room:                          # no such room → nothing to answer
        return
    join_room(room_id)                    # callee joins the room
    room['ts'] = time.time()
    room['answered'] = True
    emit('webrtc_answer', {'answer': answer}, to=room_id, include_self=False)
    for cand in room['caller_ice']:       # flush buffered caller ICE to the callee
        emit('webrtc_ice', {'candidate': cand})
    room['caller_ice'] = []
    print(f'[Call] Answer relayed for room {room_id}')


@socketio.on('webrtc_ice')
def on_webrtc_ice(data):
    data = data or {}
    room_id = data.get('room_id')
    candidate = data.get('candidate')
    if not _valid_room_id(room_id) or not candidate or _payload_too_big(candidate, _MAX_ICE_BYTES):
        return
    room = _rooms.get(room_id)
    if not room:
        return
    # buffer the caller's ICE until the callee joins, so none are dropped
    if request.sid == room['caller_sid'] and not room['answered']:
        if len(room['caller_ice']) < _MAX_ICE_PER_ROOM:
            room['caller_ice'].append(candidate)
    emit('webrtc_ice', {'candidate': candidate}, to=room_id, include_self=False)


@socketio.on('end_call')
def on_end_call(data):
    """Caller clicked End Call — reclaim the room slot immediately."""
    data = data or {}
    room_id = data.get('room_id')
    if not _valid_room_id(room_id):
        return
    room = _rooms.get(room_id)
    if room and room['caller_sid'] == request.sid:
        _rooms.pop(room_id, None)
        print(f'[Call] Room {room_id} dropped (ended)')


# -----------------------------------------------------------------------
# Video processing (frame-by-frame via the existing pipeline)
# -----------------------------------------------------------------------

def _process_video_job(job_id, input_path, output_path):
    """Background worker: decode video, swap each frame, re-encode to MP4."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        _set_job(job_id, status='error',
                 error='Could not read that video file.')
        _safe_remove(input_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 120:
        fps = 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = None
    out_w = out_h = 0
    frames_done = 0
    faces_found = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            enc_ok, buf = cv2.imencode('.jpg', frame)
            if not enc_ok:
                continue

            result_bytes, face_found, _mode = run_swap(buf.tobytes())
            arr = np.frombuffer(result_bytes, np.uint8)
            out_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if out_frame is None:
                out_frame = frame

            if writer is None:
                out_h, out_w = out_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps,
                                         (out_w, out_h))
            elif (out_frame.shape[1], out_frame.shape[0]) != (out_w, out_h):
                out_frame = cv2.resize(out_frame, (out_w, out_h))

            writer.write(out_frame)
            frames_done += 1
            if face_found:
                faces_found += 1

            if total > 0:
                progress = min(99.0, frames_done / total * 100.0)
            else:
                progress = min(99.0, frames_done * 0.5)
            _set_job(job_id, status='processing', progress=progress,
                     frames=frames_done, faces=faces_found)

            # Yield to the eventlet loop so status polls stay responsive.
            socketio.sleep(0)
    except Exception as e:  # noqa: BLE001
        print(f'[Video] job {job_id} error: {e}')
        import traceback
        traceback.print_exc()
        _set_job(job_id, status='error', error=f'Processing failed: {e}')
        cap.release()
        if writer is not None:
            writer.release()
        _safe_remove(input_path)
        return
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        _safe_remove(input_path)

    if frames_done == 0:
        _set_job(job_id, status='error',
                 error='No frames could be read from that video.')
        _safe_remove(output_path)
        return

    _set_job(job_id, status='done', progress=100.0,
             frames=frames_done, faces=faces_found)


def _safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


@app.route('/api/process_video', methods=['POST'])
def api_process_video():
    """Accept an uploaded video, start frame-by-frame swap in the background."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    file = request.files['video']
    filename = file.filename or 'video'
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        return jsonify({'error': f'Unsupported video type: {ext or "unknown"}. '
                                 'Use MP4, WebM, AVI, or MOV.'}), 415

    job_id = uuid.uuid4().hex
    input_path = os.path.join(_VIDEO_DIR, f'{job_id}_in{ext}')
    output_path = os.path.join(_VIDEO_DIR, f'{job_id}_out.mp4')
    file.save(input_path)

    if os.path.getsize(input_path) == 0:
        _safe_remove(input_path)
        return jsonify({'error': 'Empty file'}), 400

    _set_job(job_id, status='processing', progress=0.0, frames=0, faces=0,
             output_path=output_path)
    socketio.start_background_task(_process_video_job, job_id,
                                   input_path, output_path)
    return jsonify({'job_id': job_id})


@app.route('/api/process_video/status/<job_id>')
def api_process_video_status(job_id):
    job = _get_job(job_id)
    if job is None:
        return jsonify({'error': 'Unknown job'}), 404
    return jsonify({
        'status': job.get('status'),
        'progress': job.get('progress', 0.0),
        'frames': job.get('frames', 0),
        'faces': job.get('faces', 0),
        'error': job.get('error'),
    })


@app.route('/api/process_video/download/<job_id>')
def api_process_video_download(job_id):
    job = _get_job(job_id)
    if job is None or job.get('status') != 'done':
        return jsonify({'error': 'Result not ready'}), 404
    output_path = job.get('output_path')
    if not output_path or not os.path.exists(output_path):
        return jsonify({'error': 'Result file missing'}), 404
    return send_file(output_path, mimetype='video/mp4', as_attachment=True,
                     download_name='swapped.mp4')


# -----------------------------------------------------------------------
# WebSocket — live frame processing
# -----------------------------------------------------------------------

@socketio.on('frame')
def handle_frame(data):
    """Client sends the JPEG frame as raw binary (ArrayBuffer) — or, for older
    clients, {image: <base64 JPEG>}. Reply frame_result {image: <binary JPEG>,
    face_found, mode}. Binary drops the ~33% base64 tax in BOTH directions and
    the sync encode/decode that bloated the mobile round-trip."""
    try:
        if isinstance(data, (bytes, bytearray)):
            jpg_bytes = bytes(data)
        elif isinstance(data, dict):
            b64 = data.get('image', '')
            if ',' in b64:
                b64 = b64.split(',', 1)[1]
            jpg_bytes = base64.b64decode(b64) if b64 else b''
        else:
            return
        if not jpg_bytes:
            return
        result_bytes, face_found, mode = run_swap(jpg_bytes)
        emit('frame_result', {'image': result_bytes, 'face_found': face_found, 'mode': mode})
    except Exception as e:
        print(f'[WS] frame error: {e}')
        emit('frame_error', {'message': str(e)})


@socketio.on('connect')
def on_connect():
    print(f'[WS] Client connected: {request.sid}')


@socketio.on('disconnect')
def on_disconnect():
    _drop_rooms_for_sid(request.sid)
    print(f'[WS] Client disconnected: {request.sid}')


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == '__main__':
    print('Starting DeepFaceLive Web on http://0.0.0.0:5000')
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
