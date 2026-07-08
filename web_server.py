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
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

import cv2
import numpy as np

from datetime import timedelta

from flask import (Flask, jsonify, make_response, redirect, render_template,
                   request, send_file, session, url_for)
from flask_socketio import SocketIO, emit, join_room
from werkzeug.utils import secure_filename

import auth
import telegram_bot

# --- GPU: make onnxruntime-gpu locate the CUDA/cuDNN shared objects that ship
# in the pip nvidia-* packages. Without this, onnxruntime-gpu on Colab crashes
# at import with "libcudnn.so.9 ... cannot open shared object file" instead of
# using the GPU. preload_dlls() exists only in the GPU build (>=1.21), so this
# is a harmless no-op on the CPU-only Replit host.
try:
    import onnxruntime as _ort
    if hasattr(_ort, "preload_dlls"):
        _ort.preload_dlls()
        print("[gpu] onnxruntime CUDA/cuDNN libraries preloaded.")
except Exception as _ort_e:  # never let GPU setup stop the server from booting
    print("[gpu] onnxruntime preload skipped:", _ort_e)

from web_pipeline import FaceSwapPipeline, MODELS_DIR

app = Flask(__name__)
# Session cookies are signed with this key. When access gating is on (Replit
# host with DATABASE_URL) a strong SESSION_SECRET is REQUIRED — a known fallback
# would let anyone forge a "paid" session. On the open Colab clone (no gating)
# a fallback is harmless.
_secret = os.environ.get('SESSION_SECRET')
if not _secret:
    if auth.enabled():
        raise RuntimeError('SESSION_SECRET is required when access-code gating '
                           'is enabled (DATABASE_URL set); refusing insecure fallback.')
    _secret = 'deepfacelive-open-clone'
app.config['SECRET_KEY'] = _secret
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB upload cap (DFM models can be large)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=366)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet',
                    max_http_buffer_size=10 * 1024 * 1024)

ALLOWED_IMAGE_MIMES = {'image/jpeg', 'image/png', 'image/webp', 'image/gif'}
ALLOWED_VIDEO_EXTS = {'.mp4', '.webm', '.avi', '.mov', '.mkv', '.m4v'}

pipeline = FaceSwapPipeline()


# -----------------------------------------------------------------------
# Access gating (subscription codes) — ACTIVE ONLY when DATABASE_URL is set.
# On the free Colab GPU clone there is no DB, so the app runs fully open and
# the smooth-GPU path is never blocked. On the always-on Replit host the gate
# is on: visitors must log in with a valid access code before using the swap.
#
# Public (no code needed): the landing/login pages, static assets, the socket
# handshake, and the call *guest* surface (/call/<id> + /api/room/<id>) so the
# person you're calling can always join a shared link.
# -----------------------------------------------------------------------
PAYMENT_INFO = {
    'accounts': [
        {'bank': 'Opay', 'number': '9132883869'},
        {'bank': 'Moniepoint', 'number': '9132883869'},
    ],
    'whatsapp': '2349132883869',        # buyers send proof of payment here
    'whatsapp_display': '0913 288 3869',
}

_PUBLIC_PREFIXES = ('/static/', '/call/', '/api/room/', '/socket.io')
_PUBLIC_PATHS = {'/', '/welcome', '/login', '/logout', '/favicon.ico'}


def _session_valid():
    try:
        exp = session.get('access_exp')
        return bool(exp) and float(exp) > time.time()
    except Exception:
        return False


def _is_public(path):
    return path in _PUBLIC_PATHS or path.startswith(_PUBLIC_PREFIXES)


@app.before_request
def _access_gate():
    if not auth.enabled():
        return  # no DB → run open (Colab / local dev)
    if _is_public(request.path) or _session_valid():
        return
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Access code required', 'auth': False}), 401
    return redirect(url_for('welcome'))


def _read_device_id():
    """Return the browser's existing device id only if it looks like one we
    issued (32 hex chars); otherwise None."""
    d = request.cookies.get('tmfg_device') or ''
    return d if re.fullmatch(r'[0-9a-f]{32}', d) else None


def _set_device_cookie(resp, device_id):
    # Mark the cookie Secure when the browser reached us over HTTPS (the Replit
    # proxy sets X-Forwarded-Proto); left off for plain-http local dev so
    # testing still works.
    xfp = request.headers.get('X-Forwarded-Proto', '').split(',')[0].strip()
    secure = request.is_secure or xfp == 'https'
    resp.set_cookie('tmfg_device', device_id, max_age=400 * 24 * 3600,
                    httponly=True, samesite='Lax', secure=secure)
    return resp


def _landing(error=None, status=200):
    """Render the access-code landing page and, crucially, hand the browser its
    device id BEFORE it ever redeems a code. A code binds to the first device
    that redeems it; if that id were minted only during the login POST, a dropped
    response, a blocked cookie, or a double-tapped submit could bind the code to
    an id the browser never keeps — locking the code as "in use" forever. Issuing
    it here means the id is already accepted by the time a code is redeemed."""
    resp = make_response(render_template('landing.html', tiers=auth.TIERS_LIST,
                                         pay=PAYMENT_INFO, error=error), status)
    if not _read_device_id():
        _set_device_cookie(resp, secrets.token_hex(16))
    return resp


@app.route('/welcome')
def welcome():
    if auth.enabled() and _session_valid():
        return redirect(url_for('index'))
    return _landing()


@app.route('/login', methods=['POST'])
def login():
    if not auth.enabled():
        return redirect(url_for('index'))
    code = (request.form.get('code') or '').strip()
    # Reuse the id the landing page already handed this browser; only mint a new
    # one as a last resort (e.g. a POST that skipped /welcome). Because the id is
    # already an accepted cookie, redeeming twice (double-tap) binds to the same
    # device and stays idempotent instead of eating the code.
    device_id = _read_device_id() or secrets.token_hex(16)
    res = auth.redeem(code, device_id)
    if res.get('ok'):
        exp = res['expires_at']
        session.permanent = True
        session['access_code'] = code.upper()
        session['access_tier'] = res['tier']
        session['access_exp'] = exp.timestamp() if hasattr(exp, 'timestamp') else float(exp)
        return _set_device_cookie(redirect(url_for('index')), device_id)
    reasons = {
        'invalid': 'That access code is not valid.',
        'expired': 'That access code has expired.',
        'revoked': 'That access code has been revoked.',
        'in_use': 'This code is already in use on another device. '
                  'Each code works on one device only.',
        'empty': 'Please enter your access code.',
    }
    # Keep the same device id on the retry page so the next attempt reuses it.
    resp = make_response(render_template('landing.html', tiers=auth.TIERS_LIST,
                         pay=PAYMENT_INFO,
                         error=reasons.get(res.get('reason'), 'Could not log in.')), 401)
    return _set_device_cookie(resp, device_id)


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    return redirect(url_for('welcome'))


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

# Processed videos are throwaway artifacts: keep them only long enough for the
# user to download, then reclaim the disk. Without this, every {job_id}_out.mp4
# lived in the temp dir forever and slowly filled the host.
_VIDEO_TTL_SECONDS = 60 * 60                     # discard renders older than 1h
_VIDEO_DIR_MAX_BYTES = 2 * 1024 * 1024 * 1024    # hard cap: 2 GB, oldest-first


def _set_job(job_id, **fields):
    with _video_jobs_lock:
        job = _video_jobs.setdefault(job_id, {})
        # Stamp the completion time once, so record retention is measured from
        # when the job FINISHED rather than when it was uploaded.
        if fields.get('status') in ('done', 'error') and 'finished_at' not in job:
            fields.setdefault('finished_at', time.time())
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
    # Unauthenticated visitors — and the deployment's "/" healthcheck — get the
    # landing page with a 200 here instead of a redirect, since some deploy
    # healthcheckers only treat a 2xx on "/" as healthy. Authenticated users get
    # the app itself.
    if auth.enabled() and not _session_valid():
        return _landing()
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
    """Background worker: decode video, swap each frame to a SILENT render, then
    mux the original audio back in so the download keeps its sound."""
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

    # OpenCV's VideoWriter can only write video, so frames go to a silent temp
    # file first; the original audio is muxed in afterwards with ffmpeg.
    silent_path = os.path.join(_VIDEO_DIR, f'{job_id}_silent.mp4')
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
                writer = cv2.VideoWriter(silent_path, fourcc, fps,
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
        _safe_remove(silent_path)
        return
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if frames_done == 0:
        _set_job(job_id, status='error',
                 error='No frames could be read from that video.')
        _safe_remove(input_path)
        _safe_remove(silent_path)
        return

    # Re-attach the original soundtrack. The input file is still needed here as
    # the audio source, so it's only cleaned up after muxing.
    try:
        _mux_original_audio(silent_path, input_path, output_path)
    finally:
        _safe_remove(input_path)
        _safe_remove(silent_path)

    if not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
        _set_job(job_id, status='error',
                 error='Could not finalize the swapped video.')
        return

    _set_job(job_id, status='done', progress=100.0,
             frames=frames_done, faces=faces_found)


def _safe_remove(path):
    """Delete a file if present. Returns True when the path is gone afterwards
    (deleted or never existed), False only if the unlink failed."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
        return True
    except OSError:
        return False


def _mux_original_audio(silent_path, source_path, output_path):
    """Combine the swapped (silent) video with the audio from the original
    upload into output_path. ffmpeg copies the video stream untouched and only
    encodes audio, so it's fast. Falls back to the silent render (copied to
    output_path) when the source has no audio or ffmpeg is unavailable, so the
    job still yields a downloadable file either way.

    subprocess is eventlet-green here (monkey_patched), so the wait yields to the
    event loop and ffmpeg's work runs in a separate process — no server hang."""
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', silent_path,          # 0: swapped video (no audio)
        '-i', source_path,          # 1: original upload (for its audio)
        '-map', '0:v:0',            # video from the swap
        '-map', '1:a:0?',           # audio from the original ('?' = optional)
        '-c:v', 'copy',             # keep the swapped video as-is (fast)
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest', '-movflags', '+faststart',
        output_path,
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, timeout=600)
        if (proc.returncode == 0 and os.path.exists(output_path)
                and os.path.getsize(output_path) > 0):
            return
        detail = (proc.stderr or b'').decode('utf-8', 'replace').strip()[:300]
        print(f'[Video] audio mux failed (rc={proc.returncode}); '
              f'serving silent video. ffmpeg: {detail}')
    except Exception as e:  # noqa: BLE001 — ffmpeg missing, timeout, etc.
        print(f'[Video] audio mux error ({e}); serving silent video.')

    # Fallback: hand back the silent render so the download still works.
    try:
        _safe_remove(output_path)
        shutil.copyfile(silent_path, output_path)
    except OSError as e:
        print(f'[Video] fallback copy failed: {e}')


def _cleanup_video_dir():
    """Reclaim disk from finished/orphaned video files. Deletes anything past
    the TTL, then — if the folder is still over its size cap — removes the oldest
    files until it fits, and finally prunes stale job records. Runs on every new
    upload and on a periodic sweep. Best-effort; never raises."""
    now = time.time()
    # Never touch files belonging to a job that is still uploading or running.
    with _video_jobs_lock:
        active = tuple(jid for jid, job in _video_jobs.items()
                       if job.get('status') in ('uploading', 'processing'))
    try:
        entries = []
        for name in os.listdir(_VIDEO_DIR):
            if active and name.startswith(active):
                continue
            path = os.path.join(_VIDEO_DIR, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            if os.path.isfile(path):
                entries.append((path, st.st_mtime, st.st_size))
    except OSError:
        return

    survivors = []
    for path, mtime, size in entries:
        if now - mtime > _VIDEO_TTL_SECONDS:
            _safe_remove(path)
        else:
            survivors.append((path, mtime, size))

    # Size cap: drop oldest renders until under the limit. Only count space as
    # freed when the delete actually succeeds, to avoid accounting drift.
    total = sum(size for _, _, size in survivors)
    if total > _VIDEO_DIR_MAX_BYTES:
        for path, _mtime, size in sorted(survivors, key=lambda e: e[1]):
            if total <= _VIDEO_DIR_MAX_BYTES:
                break
            if _safe_remove(path):
                total -= size

    # Prune job records once their result is gone. Retention is anchored to when
    # the job FINISHED (not when it was uploaded), so a long (>TTL) render isn't
    # dropped the instant it completes. Never prune an uploading/running job.
    with _video_jobs_lock:
        for job_id in list(_video_jobs.keys()):
            job = _video_jobs[job_id]
            if job.get('status') in ('uploading', 'processing'):
                continue
            out = job.get('output_path')
            if out and os.path.exists(out):
                continue  # result still downloadable — keep the record
            if job.get('status') == 'done' or \
                    (now - job.get('finished_at', now) > _VIDEO_TTL_SECONDS):
                _video_jobs.pop(job_id, None)


def _video_janitor():
    """Periodic sweep so renders are reclaimed even without new uploads (e.g. a
    user downloads once and leaves)."""
    while True:
        socketio.sleep(600)  # every 10 minutes
        _cleanup_video_dir()


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

    # Reclaim disk from previous jobs before writing a new (possibly large) file.
    _cleanup_video_dir()

    job_id = uuid.uuid4().hex
    input_path = os.path.join(_VIDEO_DIR, f'{job_id}_in{ext}')
    output_path = os.path.join(_VIDEO_DIR, f'{job_id}_out.mp4')
    # Register BEFORE writing the file so a concurrent cleanup never mistakes
    # this job's temp files for orphans (eventlet yields mid-save on big uploads).
    _set_job(job_id, status='uploading', progress=0.0, frames=0, faces=0,
             output_path=output_path, created=time.time())
    file.save(input_path)

    if os.path.getsize(input_path) == 0:
        _safe_remove(input_path)
        with _video_jobs_lock:
            _video_jobs.pop(job_id, None)
        return jsonify({'error': 'Empty file'}), 400

    _set_job(job_id, status='processing')
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
    if auth.enabled() and not _session_valid():
        emit('frame_error', {'message': 'Session expired — please log in again.', 'auth': False})
        return
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
    if auth.enabled():
        try:
            auth.init_db()
            print('[Auth] Access-code gating ENABLED (DATABASE_URL present).')
        except Exception as e:  # noqa: BLE001
            print(f'[Auth] init_db failed: {e}')
        if telegram_bot.enabled():
            # The bot writes codes into whatever DB this process is connected to.
            # In development that's the DEV database, but the PUBLISHED app reads
            # the PRODUCTION database — so a dev bot creates codes the live app can
            # never see ("invalid code"). Only run the bot where it writes to the
            # same DB the live app reads: the Replit deployment (REPLIT_DEPLOYMENT
            # is set there), or any host where we opt in with RUN_TELEGRAM_BOT=1.
            run_bot = bool(os.environ.get('REPLIT_DEPLOYMENT')) or \
                os.environ.get('RUN_TELEGRAM_BOT') == '1'
            if run_bot:
                eventlet.spawn(telegram_bot.run)
                print('[Bot] Telegram code bot started — codes go to THIS '
                      'environment\'s database.')
            else:
                print('[Bot] Telegram bot idle in development. It runs only in the '
                      'published app, so generated codes land in the PRODUCTION '
                      'database the live site reads. Set RUN_TELEGRAM_BOT=1 to force '
                      'it here for local testing.')
        else:
            print('[Bot] Telegram bot idle — set TELEGRAM_BOT_TOKEN + '
                  'BOT_ADMIN_PASSPHRASE to enable code generation.')
    else:
        print('[Auth] No DATABASE_URL — running OPEN, no access gate '
              '(expected on the Colab GPU clone).')
    # Bind to the host-provided port. Railway / Cloud Run inject $PORT (e.g. 8080);
    # Replit and the Colab clone leave it unset, so we fall back to 5000.
    _port = int(os.environ.get('PORT', 5000))
    # Background sweeper so processed videos don't accumulate on disk.
    socketio.start_background_task(_video_janitor)
    print(f'Starting DeepFaceLive Web on http://0.0.0.0:{_port}')
    socketio.run(app, host='0.0.0.0', port=_port, debug=False)
