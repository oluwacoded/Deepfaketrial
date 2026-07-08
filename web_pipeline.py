"""
DeepFaceLive Web Pipeline
Wraps the existing backend ML code for use in a headless web server.
No Qt, no multiprocessing – single-threaded CPU inference.
"""
import base64
import io
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


def _pick_device():
    """Pick the best available inference device.

    Auto-detects at runtime: on a GPU host (e.g. Google Colab with
    onnxruntime-gpu installed) this returns the CUDA device so inference runs
    fast; on a CPU-only host (e.g. Replit) it returns CPU unchanged. Set the
    FORCE_CPU=1 environment variable to always use CPU.
    """
    if os.environ.get('FORCE_CPU') == '1':
        print('[Pipeline] FORCE_CPU set - using CPU device.')
        return _cpu_device
    try:
        for dev in lib_ort.get_available_devices_info(include_cpu=False):
            if not dev.is_cpu():
                print(f'[Pipeline] GPU detected - using {dev}')
                return dev
    except Exception as e:
        print(f'[Pipeline] GPU detection failed ({e}); falling back to CPU.')
    print('[Pipeline] No GPU detected - using CPU device.')
    return _cpu_device


_device = _pick_device()
_device_lock = threading.Lock()


class FaceSwapPipeline:
    """
    Simplified, single-threaded DeepFaceLive pipeline for web use.
    Detection: YoloV5Face (bundled ONNX, no download needed)
    Swap:      DFMModel (.dfm celebrity models)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._detector: Optional[YoloV5Face] = None
        self._dfm_model: Optional[DFMModel] = None
        self._current_model_name: Optional[str] = None
        self._model_loading = False
        self._model_load_progress: float = 0.0  # 0–100
        self._model_load_error: Optional[str] = None
        self._enabled = True

        # Merge params (match FaceMerger defaults)
        self.face_coverage = 2.2
        self.face_output_size = 192
        self.morph_factor = 0.75
        self.face_opacity = 1.0
        self.erode_amount = 5
        self.blur_amount = 25
        self.color_transfer = 'rct'  # 'rct' or 'none'

        self._init_detector()

    # ------------------------------------------------------------------
    # Detector initialisation
    # ------------------------------------------------------------------

    def _init_detector(self):
        global _device
        try:
            self._detector = YoloV5Face(_device)
            where = 'CPU' if _device.is_cpu() else f'GPU ({_device})'
            print(f'[Pipeline] YoloV5Face detector ready on {where}.')
        except Exception as e:
            print(f'[Pipeline] Detector init failed on {_device}: {e}')
            if not _device.is_cpu():
                print('[Pipeline] Falling back to CPU...')
                with _device_lock:
                    _device = _cpu_device
                try:
                    self._detector = YoloV5Face(_device)
                    print('[Pipeline] YoloV5Face detector ready on CPU (fallback).')
                except Exception as e2:
                    print(f'[Pipeline] CPU fallback also failed: {e2}')

    # ------------------------------------------------------------------
    # Model catalogue
    # ------------------------------------------------------------------

    @staticmethod
    def get_available_models():
        infos = get_available_models_info(MODELS_DIR)
        result = []
        for info in infos:
            result.append({
                'name': info.get_name(),
                'downloaded': info.get_model_path().exists(),
                'has_url': info.get_url() is not None,
            })
        return result

    # ------------------------------------------------------------------
    # Model loading (with optional download)
    # ------------------------------------------------------------------

    def load_model(self, model_name: str):
        """Start loading a model in a background thread."""
        with self._lock:
            if self._model_loading:
                return
            self._model_loading = True
            self._model_load_progress = 0.0
            self._model_load_error = None

        threading.Thread(target=self._load_model_thread,
                         args=(model_name,), daemon=True).start()

    def _load_model_thread(self, model_name: str):
        from modelhub.DFLive.DFMModel import get_available_models_info

        try:
            infos = get_available_models_info(MODELS_DIR)
            info: Optional[DFMModelInfo] = None
            for i in infos:
                if i.get_name() == model_name:
                    info = i
                    break

            if info is None:
                with self._lock:
                    self._model_load_error = f'Model "{model_name}" not found.'
                return

            device = _device
            dfm_model, error = self._run_dfm_initializer(info, device)

            # If the model failed to initialise on the GPU, retry this one load on
            # CPU so face-swap still works (e.g. onnxruntime-gpu / CUDA mismatch on
            # Colab). We do NOT permanently demote the global device here: a
            # transient download/model error must not force later loads onto CPU,
            # and a truly broken GPU is already caught by the detector at startup.
            if error is not None and not device.is_cpu():
                print(f'[Pipeline] Model load failed on GPU ({error}); retrying on CPU...')
                with self._lock:
                    self._model_load_progress = 0.0
                device = _cpu_device
                dfm_model, error = self._run_dfm_initializer(info, device)

            if dfm_model is not None:
                where = 'CPU' if device.is_cpu() else f'GPU ({device})'
                with self._lock:
                    self._dfm_model = dfm_model
                    self._current_model_name = model_name
                    self._model_load_progress = 100.0
                print(f'[Pipeline] Model "{model_name}" loaded on {where}.')
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

    def _run_dfm_initializer(self, info, device):
        """Run a DFMModelInitializer to completion on the given device.

        Returns (dfm_model, error) where exactly one is non-None. Handles the
        download + init event loop and updates load progress along the way.
        """
        from modelhub.DFLive.DFMModel import DFMModelInitializer

        initializer = DFMModelInitializer(info, device)
        while True:
            events = initializer.process_events()
            if events.new_status_downloading or events.prev_status_downloading:
                p = events.download_progress
                if p is not None:
                    with self._lock:
                        self._model_load_progress = p
            if events.new_status_initialized:
                return events.dfm_model, None
            if events.new_status_error:
                return None, events.error
            time.sleep(0.1)

    def get_model_status(self):
        with self._lock:
            return {
                'current': self._current_model_name,
                'loading': self._model_loading,
                'progress': self._model_load_progress,
                'error': self._model_load_error,
            }

    # ------------------------------------------------------------------
    # Frame processing — returns (jpeg_bytes, face_found)
    # ------------------------------------------------------------------

    def process_frame(self, jpg_bytes: bytes) -> Tuple[bytes, bool]:
        """
        Accept raw JPEG bytes, run face-swap, return (result_jpeg, face_found).
        face_found=False means no face was detected (or no model loaded).
        """
        # Decode
        arr = np.frombuffer(jpg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR uint8 HWC

        if frame is None:
            return jpg_bytes, False

        # Clamp huge frames (memory / CPU exhaustion)
        max_dim = 1280
        h, w = frame.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        with self._lock:
            detector = self._detector
            dfm_model = self._dfm_model
            enabled = self._enabled

        if not enabled or detector is None or dfm_model is None:
            return self._encode(frame), False

        try:
            result, face_found = self._swap_faces(frame, detector, dfm_model)
        except Exception as e:
            print(f'[Pipeline] process_frame error: {e}')
            import traceback; traceback.print_exc()
            result, face_found = frame, False

        return self._encode(result), face_found

    def _swap_faces(self, frame_bgr, detector, dfm_model) -> Tuple[np.ndarray, bool]:
        """Run detection → align → swap → merge for one frame."""
        H, W = frame_bgr.shape[:2]

        # Convert to RGB for processing (models expect RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 1. Detect faces — lower threshold for dark / side-lit faces
        faces_per_batch = detector.extract(
            frame_rgb, threshold=0.25, fixed_window=640)
        if not faces_per_batch or not faces_per_batch[0]:
            return frame_bgr, False

        # Process first detected face only (largest)
        detections = faces_per_batch[0]
        detections = sorted(detections, key=lambda d: (d[2]-d[0])*(d[3]-d[1]), reverse=True)
        l, t, r, b = detections[0]

        # 2. Build FRect (uniform coordinates)
        face_rect = FRect.from_ltrb((l / W, t / H, r / W, b / H))

        # 3. Cut & align the face
        face_align_img, uni_mat = face_rect.cut(
            frame_rgb,
            coverage=self.face_coverage,
            output_size=self.face_output_size,
        )
        face_h, face_w = face_align_img.shape[:2]

        # 4. DFM swap
        out_celeb, out_celeb_mask, out_face_mask = dfm_model.convert(
            face_align_img, morph_factor=self.morph_factor)

        # Outputs are NHWC; squeeze batch dim
        celeb_img  = out_celeb[0]       # HWC float32
        celeb_mask = out_celeb_mask[0]  # HW1 float32
        face_mask  = out_face_mask[0]   # HW1 float32

        # 5. Build inverse transform: aligned → source frame
        aligned_to_source = uni_mat.invert().to_exact_mat(
            face_w, face_h, W, H)

        # 6. Merge back onto the frame
        merged = self._merge_cpu(
            frame_rgb,
            face_align_img,
            celeb_img,
            celeb_mask,
            face_mask,
            aligned_to_source,
            W, H, face_w, face_h,
        )

        # Convert result float32 HWC [0,1] → uint8 → BGR
        merged_uint8 = np.clip(merged * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(merged_uint8, cv2.COLOR_RGB2BGR), True

    def _merge_cpu(self, frame_rgb, face_align_img, celeb_img, celeb_mask,
                   face_mask, aligned_to_source_mat, W, H, face_w, face_h):
        """
        Pure-numpy / ImageProcessor merge (mirrors FaceMerger._merge_on_cpu).
        All images are float32 in [0,1] range unless stated.
        """
        import numexpr as ne

        frame_f = ImageProcessor(frame_rgb).to_ufloat32().get_image('HWC')

        # Build combined mask (face_mask * celeb_mask)
        fm = ImageProcessor(face_mask).to_ufloat32().get_image('HW')
        cm = ImageProcessor(celeb_mask).to_ufloat32().get_image('HW')
        combined_mask = fm * cm  # HW

        # Erode & blur the mask, fade to border
        combined_mask_ip = ImageProcessor(combined_mask).erode_blur(
            self.erode_amount, self.blur_amount, fade_to_border=True)
        combined_mask_hwc = combined_mask_ip.get_image('HWC')  # HW1 float32

        # Color-match celeb face to source face tone
        celeb_ip = ImageProcessor(celeb_img).to_ufloat32()
        if self.color_transfer == 'rct':
            face_align_f = ImageProcessor(face_align_img).to_ufloat32().get_image('HWC')
            celeb_ip = celeb_ip.rct(
                like=face_align_f,
                mask=combined_mask_hwc,
                like_mask=combined_mask_hwc,
            )

        # Warp mask and celeb face back to frame space
        frame_mask = ImageProcessor(combined_mask_hwc).warp_affine(
            aligned_to_source_mat, W, H
        ).clip2(1.0 / 255.0, 0.0, 1.0, 1.0).get_image('HWC')

        frame_celeb = celeb_ip.warp_affine(
            aligned_to_source_mat, W, H,
            interpolation=ImageProcessor.Interpolation.LINEAR,
        ).get_image('HWC')

        # Blend
        opacity = np.float32(self.face_opacity)
        one_f = np.float32(1.0)
        if opacity == 1.0:
            merged = ne.evaluate(
                'frame_f*(one_f-frame_mask) + frame_celeb*frame_mask')
        else:
            merged = ne.evaluate(
                'frame_f*(one_f-frame_mask) + frame_f*frame_mask*(one_f-opacity) + frame_celeb*frame_mask*opacity')

        return merged  # HWC float32

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def unload_model(self):
        with self._lock:
            self._dfm_model = None
            self._current_model_name = None
            self._model_load_error = None
