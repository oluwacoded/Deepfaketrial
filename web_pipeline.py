"""
DeepFaceLive Web Pipeline
Wraps the existing backend ML code for use in a headless web server.

Threading model:
  - Heavy inference runs in a worker thread (via eventlet.tpool in web_server),
    so it never blocks the async event loop (that was the real cause of hangs).
  - snapshot() grabs the current model refs under a lock in the caller's
    (green) thread; run() then does pure CPU work with NO locks, making it
    safe to execute in a native worker thread.

Two swap modes:
  - DFM   : celebrity .dfm models (face replacement)
  - Paste : user-supplied photo, alpha-blended over the detected head
Detection is rotation-augmented (0/90/-90) so it still works when the
camera/user is sideways (e.g. lying down).
"""
import os
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from modelhub.DFLive.DFMModel import (DFMModel, DFMModelInfo,
                                       get_available_models_info)
from modelhub.onnx.YoloV5Face.YoloV5Face import YoloV5Face
from xlib import onnxruntime as lib_ort
from xlib.face import FRect
from xlib.image import ImageProcessor


MODELS_DIR = Path(__file__).parent / 'userdata' / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

_cpu_device = lib_ort.get_cpu_device_info()

# cv2 rotation constants for rotation-augmented detection
_ROT = {90: cv2.ROTATE_90_CLOCKWISE, -90: cv2.ROTATE_90_COUNTERCLOCKWISE}
_ROT_INV = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, -90: cv2.ROTATE_90_CLOCKWISE}


def _pick_device():
    """Prefer a GPU execution provider if one is available, else CPU.

    Set FORCE_CPU=1 to always use CPU (handy for debugging on a GPU host).
    """
    if os.environ.get('FORCE_CPU') == '1':
        print('[Pipeline] FORCE_CPU set - using CPU inference.')
        return _cpu_device
    try:
        devices = lib_ort.get_available_devices_info(include_cpu=False)
    except Exception as e:
        print(f'[Pipeline] GPU probe failed ({e}); using CPU.')
        devices = []
    if devices:
        dev = devices[0]
        print(f'[Pipeline] Using GPU device: {dev}')
        return dev
    print('[Pipeline] No GPU found; using CPU inference.')
    return _cpu_device


_device = _pick_device()
_device_lock = threading.Lock()


def _sess_providers(sess):
    """Return (on_gpu, providers) for an onnxruntime InferenceSession, safely.
    ONNX Runtime silently appends CPU and drops a GPU EP that failed to load, so
    get_providers() is the source of truth for what is ACTUALLY running."""
    try:
        provs = list(sess.get_providers())
    except Exception:
        return False, []
    on_gpu = any(p in ('CUDAExecutionProvider', 'DmlExecutionProvider') for p in provs)
    return on_gpu, provs


class FaceSwapPipeline:
    def __init__(self):
        self._lock = threading.RLock()
        self._detector: Optional[YoloV5Face] = None
        self._dfm_model: Optional[DFMModel] = None
        self._current_model_name: Optional[str] = None
        self._model_loading = False
        self._model_load_progress: float = 0.0
        self._model_load_error: Optional[str] = None
        self._enabled = True
        # Actual execution provider in use (None until probed). ONNX Runtime can
        # silently run on CPU even on a GPU host, so we record the real answer.
        self._detector_on_gpu: Optional[bool] = None
        self._model_on_gpu: Optional[bool] = None
        # Rate-limited swap heartbeat counters (for diagnosing remote runs)
        self._frame_log_n = 0
        self._frame_swapped = 0

        # Custom target face (paste mode) — BGR crop of the uploaded face+head
        self._target_face_bgr: Optional[np.ndarray] = None

        # Rotation the last successful detection used — try it first next frame
        self._preferred_rot = 0

        # Merge params
        self.face_coverage = 1.5      # tighter crop -> more real face fills the 224 model
                                      #   -> much sharper, and kills the ghosted/melted look
        self.face_output_size = 320   # cut & merge above model res for a cleaner upscaled warp
        self.morph_factor = 0.75
        self.face_opacity = 1.0
        self.erode_amount = 4
        self.blur_amount = 35
        self.color_transfer = 'rct'   # 'rct' or 'none'
        self.sharpen_amount = 0.55    # unsharp on the swap; counters 224-model softness

        self._init_detector()

    # ------------------------------------------------------------------ setup
    def _init_detector(self):
        global _device
        try:
            self._detector = YoloV5Face(_device)
            where = 'CPU' if _device.is_cpu() else f'GPU ({_device})'
            print(f'[Pipeline] YoloV5Face detector ready on {where}.')
            gpu, provs = _sess_providers(self._detector._sess)
            self._detector_on_gpu = gpu
            print(f'[Pipeline] Detector execution providers: {provs}')
            if not _device.is_cpu() and not gpu:
                print('[Pipeline] !! GPU detected but ONNX Runtime fell back to CPU — '
                      'inference will be SLOW. See the ONNX Runtime warning above '
                      '(usually a CUDA/cuDNN mismatch).')
        except Exception as e:
            print(f'[Pipeline] Failed to init detector on {_device}: {e}')
            # A GPU can be detected yet still fail to initialise (e.g.
            # onnxruntime-gpu / cuDNN mismatch on Colab). Fall back to CPU so
            # the app keeps working instead of having no detector at all.
            if not _device.is_cpu():
                print('[Pipeline] Falling back to CPU for detection...')
                with _device_lock:
                    _device = _cpu_device
                try:
                    self._detector = YoloV5Face(_device)
                    self._detector_on_gpu = False
                    print('[Pipeline] YoloV5Face detector ready on CPU (fallback).')
                except Exception as e2:
                    print(f'[Pipeline] CPU fallback also failed: {e2}')

    # -------------------------------------------------------------- catalogue
    @staticmethod
    def get_available_models():
        infos = get_available_models_info(MODELS_DIR)
        return [{'name': i.get_name(),
                 'downloaded': i.get_model_path().exists(),
                 'has_url': i.get_url() is not None,
                 'custom': i.get_url() is None} for i in infos]

    # ------------------------------------------------------------ model load
    def load_model(self, model_name: str):
        with self._lock:
            if self._model_loading:
                return
            self._model_loading = True
            self._model_load_progress = 0.0
            self._model_load_error = None
        threading.Thread(target=self._load_model_thread,
                         args=(model_name,), daemon=True).start()

    def _init_dfm(self, info, device):
        """Drive a DFMModelInitializer to completion on `device`.
        Returns (dfm_model, error_str); updates download progress as it runs.
        Bails out if the load makes no forward progress for STALL_TIMEOUT
        seconds, so a dead download can't leave the model stuck 'loading'."""
        from modelhub.DFLive.DFMModel import DFMModelInitializer
        STALL_TIMEOUT = 300.0
        initializer = DFMModelInitializer(info, device)
        last_change = time.monotonic()
        last_progress = -1.0
        while True:
            events = initializer.process_events()
            # Update progress on ANY reported value (not only on status flips):
            # steady DOWNLOADING events carry download_progress too.
            p = getattr(events, 'download_progress', None)
            if p is not None:
                with self._lock:
                    self._model_load_progress = p
                if p != last_progress:
                    last_progress = p
                    last_change = time.monotonic()
            if events.new_status_initialized:
                return events.dfm_model, None
            if events.new_status_error:
                return None, events.error
            if time.monotonic() - last_change > STALL_TIMEOUT:
                return None, ('Model load stalled (no progress). '
                              'Check the connection and try again.')
            time.sleep(0.1)

    def _load_model_thread(self, model_name: str):
        from modelhub.DFLive.DFMModel import get_available_models_info
        with _device_lock:
            device = _device  # snapshot once so a concurrent device swap can't split this load
        try:
            infos = get_available_models_info(MODELS_DIR)
            info = next((i for i in infos if i.get_name() == model_name), None)
            if info is None:
                with self._lock:
                    self._model_load_error = f'Model "{model_name}" not found.'
                return

            dfm_model, error = self._init_dfm(info, device)
            # Retry this one load on CPU if it failed on the GPU (e.g.
            # onnxruntime-gpu / CUDA mismatch). We do NOT demote the global
            # device: a transient download error must not force later loads to CPU.
            if error is not None and not device.is_cpu():
                print(f'[Pipeline] Model load failed on GPU ({error}); retrying on CPU...')
                with self._lock:
                    self._model_load_progress = 0.0
                dfm_model, error = self._init_dfm(info, _cpu_device)

            if dfm_model is not None:
                gpu, provs = _sess_providers(dfm_model._sess)
                with self._lock:
                    self._dfm_model = dfm_model
                    self._current_model_name = model_name
                    self._model_load_progress = 100.0
                    self._model_on_gpu = gpu
                print(f'[Pipeline] Model "{model_name}" loaded. Execution providers: {provs}')
                if not gpu and not device.is_cpu():
                    print('[Pipeline] !! Model requested GPU but is running on CPU — '
                          'DFM will be laggy. Likely a CUDA/cuDNN mismatch.')
            else:
                with self._lock:
                    self._model_load_error = error
                print(f'[Pipeline] Model load error: {error}')
        except Exception as e:
            print(f'[Pipeline] Unexpected error loading model: {e}')
            with self._lock:
                self._model_load_error = f'Unexpected error: {e}'
        finally:
            with self._lock:
                self._model_loading = False

    def unload_model(self):
        with self._lock:
            self._dfm_model = None
            self._current_model_name = None
            self._model_load_error = None
            self._model_on_gpu = None

    def get_model_status(self):
        with self._lock:
            has_target = self._target_face_bgr is not None
            mode = 'paste' if has_target else ('dfm' if self._dfm_model else 'none')
            return {
                'current': self._current_model_name,
                'loading': self._model_loading,
                'progress': self._model_load_progress,
                'error': self._model_load_error,
                'mode': mode,
                'target_face_set': has_target,
                'device': self._effective_device(),
            }

    def _effective_device(self) -> str:
        """Report the device inference is ACTUALLY on (ONNX Runtime can silently
        fall back to CPU on a GPU host, e.g. a CUDA/cuDNN mismatch)."""
        if self._dfm_model is not None and self._model_on_gpu is not None:
            on_gpu = self._model_on_gpu
        elif self._detector_on_gpu is not None:
            on_gpu = self._detector_on_gpu
        else:
            on_gpu = not _device.is_cpu()
        return 'gpu' if on_gpu else 'cpu'

    # ------------------------------------------------------- target (paste)
    def set_target_face(self, jpg_bytes: bytes) -> dict:
        arr = np.frombuffer(jpg_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {'ok': False, 'error': 'Could not decode image'}

        h, w = img.shape[:2]
        if max(h, w) > 1280:
            sc = 1280 / max(h, w)
            img = cv2.resize(img, (int(w * sc), int(h * sc)))
            h, w = img.shape[:2]

        with self._lock:
            detector = self._detector
        if detector is None:
            return {'ok': False, 'error': 'Detector not ready'}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bright = self._clahe(img_rgb)
        faces, rot = self._detect_any_rotation(detector, bright, self._preferred_rot, 0.20)
        if not faces or not faces[0]:
            return {'ok': True, 'face_found': False}

        work = cv2.rotate(img, _ROT[rot]) if rot in _ROT else img
        Wr = work.shape[1]; Hr = work.shape[0]
        dets = sorted(faces[0], key=lambda d: (d[2]-d[0])*(d[3]-d[1]), reverse=True)
        l, t, r, b = [int(x) for x in dets[0]]
        fw, fh = r - l, b - t
        # generous padding — include forehead + hair (head + hair)
        l = max(0, l - int(fw * 0.35)); r = min(Wr, r + int(fw * 0.35))
        t = max(0, t - int(fh * 0.55)); b = min(Hr, b + int(fh * 0.35))

        with self._lock:
            self._target_face_bgr = work[t:b, l:r].copy()
        print(f'[Pipeline] Target face set: {r-l}x{b-t}px')
        return {'ok': True, 'face_found': True}

    def clear_target_face(self):
        with self._lock:
            self._target_face_bgr = None
        print('[Pipeline] Target face cleared.')

    # --------------------------------------------------------- snapshot/run
    def snapshot(self) -> dict:
        """Grab current model refs under lock (call from the green thread)."""
        with self._lock:
            tf = self._target_face_bgr.copy() if self._target_face_bgr is not None else None
            return {
                'detector': self._detector,
                'dfm_model': self._dfm_model,
                'target_face': tf,
                'enabled': self._enabled,
                # snapshot ALL tuning params too, so run() reads nothing off
                # self in the worker thread (deterministic per frame)
                'preferred_rot': self._preferred_rot,
                'face_coverage': self.face_coverage,
                'face_output_size': self.face_output_size,
                'morph_factor': self.morph_factor,
                'face_opacity': self.face_opacity,
                'color_transfer': self.color_transfer,
                'erode_amount': self.erode_amount,
                'blur_amount': self.blur_amount,
                'sharpen_amount': self.sharpen_amount,
            }

    def process_frame(self, jpg_bytes: bytes) -> Tuple[bytes, bool, str]:
        """Convenience: snapshot + run inline (used off the hot path)."""
        return self.run(jpg_bytes, self.snapshot())

    def run(self, jpg_bytes: bytes, snap: dict) -> Tuple[bytes, bool, str]:
        """Pure CPU work — NO locks. Safe to run in a worker thread."""
        arr = np.frombuffer(jpg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jpg_bytes, False, 'none'

        h, w = frame.shape[:2]
        if max(h, w) > 1280:
            sc = 1280 / max(h, w)
            frame = cv2.resize(frame, (int(w * sc), int(h * sc)))

        detector = snap['detector']
        dfm_model = snap['dfm_model']
        target_face = snap['target_face']
        enabled = snap['enabled']

        if not enabled or detector is None:
            return self._encode(frame), False, 'none'

        found, mode = False, 'none'
        try:
            if target_face is not None:
                result, found, rot = self._paste_face(frame, detector, target_face, snap)
                mode = 'paste'
                if found:
                    self._preferred_rot = rot   # atomic int write — a hint for next frame
                out = self._encode(result)
            elif dfm_model is not None:
                result, found, rot = self._dfm_swap(frame, detector, dfm_model, snap)
                mode = 'dfm'
                if found:
                    self._preferred_rot = rot
                out = self._encode(result)
            else:
                out = self._encode(frame)
        except Exception as e:
            print(f'[Pipeline] run error: {e}')
            import traceback; traceback.print_exc()
            out, found, mode = self._encode(frame), False, 'none'
        self._log_frame_stats(found, mode)
        return out, found, mode

    def _log_frame_stats(self, found: bool, mode: str):
        """Rate-limited heartbeat so a remote (Colab) run explains itself in its
        own console: if `swapped` stays 0 while `passthrough` climbs, the model
        pipeline is running but no face is being detected (check detector_gpu);
        any convert errors are printed above by run()."""
        n = self._frame_log_n + 1
        self._frame_log_n = n
        if found:
            self._frame_swapped += 1
        if n % 50 == 0:
            swapped = self._frame_swapped
            print(f'[Pipeline] frames={n} swapped={swapped} passthrough={n - swapped} '
                  f'last_mode={mode} detector_gpu={self._detector_on_gpu} '
                  f'model_gpu={self._model_on_gpu}')

    # -------------------------------------------------------- detection util
    @staticmethod
    def _clahe(img_rgb: np.ndarray) -> np.ndarray:
        """Boost dark frames so detection works in poor lighting."""
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_mean = float(lab[:, :, 0].mean())
        if l_mean >= 100:
            return img_rgb
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        if l_mean < 70:
            boost = min(150 / max(l_mean, 1), 3.0)
            lab[:, :, 0] = np.clip(lab[:, :, 0].astype(np.float32) * boost, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def _detect_any_rotation(self, detector, frame_rgb_bright, preferred_rot=0, threshold=0.20):
        """Try detection at [preferred, 0, 90, -90]; return (faces, rot).
        Stateless: the caller persists the winning rotation, so worker threads
        never write shared state here."""
        h, w = frame_rgb_bright.shape[:2]
        # Detection window scales with the frame: small live frames (~320px)
        # detect at 384 instead of 640 -> far less CPU per frame at the same
        # accuracy for the large faces typical on a webcam. Photos (up to
        # 1280px) still use the full 640 window so small faces aren't missed.
        fw = int(np.clip((max(h, w) // 32) * 32, 384, 640))
        order = [preferred_rot]
        for r in (0, 90, -90):
            if r not in order:
                order.append(r)
        for rot in order:
            img = frame_rgb_bright if rot == 0 else cv2.rotate(frame_rgb_bright, _ROT[rot])
            faces = detector.extract(img, threshold=threshold, fixed_window=fw)
            if faces and faces[0]:
                return faces, rot
        return None, 0

    # -------------------------------------------------------------- DFM swap
    def _dfm_swap(self, frame_bgr, detector, dfm_model, snap) -> Tuple[np.ndarray, bool, int]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        bright = self._clahe(frame_rgb)
        faces, rot = self._detect_any_rotation(detector, bright, snap['preferred_rot'], 0.20)
        if not faces or not faces[0]:
            return frame_bgr, False, 0

        work_rgb = cv2.rotate(frame_rgb, _ROT[rot]) if rot in _ROT else frame_rgb
        H, W = work_rgb.shape[:2]

        dets = sorted(faces[0], key=lambda d: (d[2]-d[0])*(d[3]-d[1]), reverse=True)
        l, t, r, b = dets[0]
        face_rect = FRect.from_ltrb((l / W, t / H, r / W, b / H))

        face_align_img, uni_mat = face_rect.cut(
            work_rgb, coverage=snap['face_coverage'], output_size=snap['face_output_size'])
        face_h, face_w = face_align_img.shape[:2]

        out_celeb, out_celeb_mask, out_face_mask = dfm_model.convert(
            face_align_img, morph_factor=snap['morph_factor'])
        celeb_img = out_celeb[0]; celeb_mask = out_celeb_mask[0]; face_mask = out_face_mask[0]

        aligned_to_source = uni_mat.invert().to_exact_mat(face_w, face_h, W, H)
        merged = self._merge_cpu(work_rgb, face_align_img, celeb_img, celeb_mask,
                                 face_mask, aligned_to_source, W, H, face_w, face_h, snap)

        result = cv2.cvtColor(np.clip(merged * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if rot in _ROT_INV:
            result = cv2.rotate(result, _ROT_INV[rot])
        return result, True, rot

    def _merge_cpu(self, frame_rgb, face_align_img, celeb_img, celeb_mask,
                   face_mask, aligned_to_source_mat, W, H, face_w, face_h, snap):
        import numexpr as ne
        frame_f = ImageProcessor(frame_rgb).to_ufloat32().get_image('HWC')

        fm = ImageProcessor(face_mask).to_ufloat32().get_image('HW')
        cm = ImageProcessor(celeb_mask).to_ufloat32().get_image('HW')
        combined_mask = fm * cm

        combined_mask_hwc = ImageProcessor(combined_mask).erode_blur(
            snap['erode_amount'], snap['blur_amount'], fade_to_border=True).get_image('HWC')

        if snap['color_transfer'] == 'rct':
            celeb_f = ImageProcessor(celeb_img).to_ufloat32().get_image('HWC')
            face_align_f = ImageProcessor(face_align_img).to_ufloat32().get_image('HWC')
            celeb_f = self._robust_color_match(celeb_f, face_align_f, combined_mask)
            celeb_ip = ImageProcessor(celeb_f)
        else:
            celeb_ip = ImageProcessor(celeb_img).to_ufloat32()

        frame_mask = ImageProcessor(combined_mask_hwc).warp_affine(
            aligned_to_source_mat, W, H).clip2(1.0 / 255.0, 0.0, 1.0, 1.0).get_image('HWC')
        frame_celeb = celeb_ip.warp_affine(
            aligned_to_source_mat, W, H,
            interpolation=ImageProcessor.Interpolation.LANCZOS4).get_image('HWC')

        # Sharpen the upscaled swap: the 224px model output goes soft once warped
        # up to the in-frame face, which reads as a melted/ghosted 'clown' face.
        sharpen = float(snap.get('sharpen_amount', 0.0))
        if sharpen > 0.0:
            blurred = cv2.GaussianBlur(frame_celeb, (0, 0), 1.5)
            frame_celeb = np.clip(
                cv2.addWeighted(frame_celeb, 1.0 + sharpen, blurred, -sharpen, 0.0),
                0.0, 1.0)

        opacity = np.float32(snap['face_opacity']); one_f = np.float32(1.0)
        if opacity == 1.0:
            merged = ne.evaluate('frame_f*(one_f-frame_mask) + frame_celeb*frame_mask')
        else:
            merged = ne.evaluate('frame_f*(one_f-frame_mask) + frame_f*frame_mask*(one_f-opacity) + frame_celeb*frame_mask*opacity')
        return merged

    @staticmethod
    def _robust_color_match(src_rgb01, ref_rgb01, mask_hw, lo=0.95, hi=1.15):
        """Match the swapped face's color to the underlying face, robustly.
        Full mean match (kills the model's cold/blue cast) and a per-channel
        std ratio clamped tight around 1.0 — this is mean-dominant transfer:
        it keeps the model's natural facial contrast (a wide clamp flattens
        skin to a dull ashy grey, especially light models on dark subjects)
        while the tight upper bound still stops bright pixels from blowing out
        to glowing orange. Computed only over the face mask so the background
        never skews the statistics."""
        m = mask_hw > 0.3
        if int(m.sum()) < 50:
            return src_rgb01
        src8 = np.clip(src_rgb01 * 255.0, 0, 255).astype(np.uint8)
        ref8 = np.clip(ref_rgb01 * 255.0, 0, 255).astype(np.uint8)
        src_lab = cv2.cvtColor(src8, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(ref8, cv2.COLOR_RGB2LAB).astype(np.float32)
        out = src_lab.copy()
        for i in range(3):
            s = src_lab[:, :, i][m]; r = ref_lab[:, :, i][m]
            sm, ss = float(s.mean()), float(s.std()) + 1e-5
            rm, rs = float(r.mean()), float(r.std()) + 1e-5
            ratio = min(hi, max(lo, rs / ss))
            out[:, :, i] = (src_lab[:, :, i] - sm) * ratio + rm
        out = np.clip(out, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    # ------------------------------------------------------------ paste mode
    def _paste_face(self, frame_bgr, detector, target_face_bgr, snap) -> Tuple[np.ndarray, bool, int]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        bright = self._clahe(frame_rgb)
        faces, rot = self._detect_any_rotation(detector, bright, snap['preferred_rot'], 0.20)
        if not faces or not faces[0]:
            return frame_bgr, False, 0

        work = cv2.rotate(frame_bgr, _ROT[rot]) if rot in _ROT else frame_bgr
        H, W = work.shape[:2]
        dets = sorted(faces[0], key=lambda d: (d[2]-d[0])*(d[3]-d[1]), reverse=True)
        l, t, r, b = [int(x) for x in dets[0]]

        fw, fh = r - l, b - t
        # include forehead + hair to match the head-sized target crop
        l = max(0, l - int(fw * 0.35)); r = min(W, r + int(fw * 0.35))
        t = max(0, t - int(fh * 0.55)); b = min(H, b + int(fh * 0.35))
        fw, fh = r - l, b - t
        if fw < 20 or fh < 20:
            return frame_bgr, False, 0

        target = cv2.resize(target_face_bgr, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
        roi = work[t:b, l:r]
        target = self._match_color(target, roi)

        # feathered elliptical mask (fast alpha-blend)
        mask = np.zeros((fh, fw), np.uint8)
        cv2.ellipse(mask, (fw // 2, fh // 2),
                    (max(1, int(fw * 0.46)), max(1, int(fh * 0.48))), 0, 0, 360, 255, -1)
        k = max(3, (min(fw, fh) // 5) | 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)

        alpha = mask[:, :, None].astype(np.float32) / 255.0
        result = work.copy()
        result[t:b, l:r] = np.clip(target.astype(np.float32) * alpha +
                                   roi.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)
        if rot in _ROT_INV:
            result = cv2.rotate(result, _ROT_INV[rot])
        return result, True, rot

    @staticmethod
    def _match_color(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
        if src_bgr.size == 0 or ref_bgr.size == 0:
            return src_bgr
        s = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        rref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        for i in range(3):
            sm, ss = s[:, :, i].mean(), s[:, :, i].std() + 1e-6
            rm, rs = rref[:, :, i].mean(), rref[:, :, i].std() + 1e-6
            s[:, :, i] = (s[:, :, i] - sm) * (rs / ss) + rm
        return cv2.cvtColor(np.clip(s, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _encode(img_bgr: np.ndarray, quality: int = 82) -> bytes:
        ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else b''

    def set_enabled(self, v: bool):
        with self._lock:
            self._enabled = v

    def set_morph_factor(self, v: float):
        self.morph_factor = float(np.clip(v, 0.0, 1.0))

    def set_face_opacity(self, v: float):
        self.face_opacity = float(np.clip(v, 0.0, 1.0))

    def set_color_transfer(self, v: str):
        self.color_transfer = v if v in ('rct', 'none') else 'rct'

    def set_face_output_size(self, v):
        try:
            self.face_output_size = int(np.clip(int(v), 96, 320))
        except (TypeError, ValueError):
            pass
