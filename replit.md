# DeepFaceLive Web

## Project overview
DeepFaceLive is a real-time AI face-swap application originally built as a Windows desktop app with a Qt GUI. This Replit fork runs it as a **web app** — your webcam stream is processed server-side with ONNX Runtime (auto-detects a GPU on Colab, falls back to CPU on Replit) and the swapped video is streamed back to the browser.

### Architecture
```
Browser (webcam) → WebSocket → Flask/SocketIO server → Pipeline → WebSocket → Browser (output)
```

Pipeline stages (run on the GPU when available, else CPU):
1. **YoloV5Face** – face detection (bundled ONNX, no download)
2. **FRect.cut** – face crop & alignment
3. **DFMModel** – face swap (.dfm celebrity models, downloaded on demand)
4. **Merge** – blend swapped face back into frame with color transfer

### Key files
| File | Purpose |
|------|---------|
| `web_server.py` | Flask + SocketIO server, REST endpoints, WebSocket frame handler |
| `web_pipeline.py` | Headless pipeline wrapper (no Qt, single-threaded CPU) |
| `templates/index.html` | Single-page web UI |
| `main.py` | Original CLI entry point (kept intact) |
| `apps/DeepFaceLive/` | Original Qt desktop app (untouched) |
| `modelhub/` | ONNX model wrappers |
| `xlib/` | Utility library (image, face, math, etc.) |
| `userdata/models/` | Downloaded .dfm face models |

### How to run
```
python web_server.py
```
Then open the web preview. The server listens on port 5000.

### Smooth GPU version (DFM) — how to access
DFM (celebrity) mode runs an AI model on every frame; on Replit's 2-core CPU it lags (~2 FPS). For smooth DFM, run the SAME app on a free Colab T4 GPU.

Permanent link (bookmark it):
https://colab.research.google.com/github/oluwacoded/Deepfaketrial/blob/colab-gpu/deepfacelive_colab.ipynb

Steps: Runtime -> Change runtime type -> T4 GPU -> Save; run the 3 cells in order; the last cell shows a QR code + a .trycloudflare.com link -> scan the QR with your phone camera (or type the link) to open it on your phone. The Colab link is permanent; the phone link is fresh each session.

On the Replit link (CPU), use "Pick face from gallery" (paste mode) for smoothness — it skips the per-frame AI model and overrides any loaded DFM model.

### Performance
- Replit (CPU, 2 cores): DFM ~1.5-2.5 FPS (laggy); paste mode ~12x lighter and usable
- Colab (T4 GPU): DFM runs smooth (~25x faster than CPU)
- First model load: downloads ~100–200 MB .dfm file from GitHub releases
- Models are cached in `userdata/models/`

## User preferences
- Keep the original Qt desktop app code intact
- Web interface is the new entry point
- Inference auto-detects GPU (Colab) and falls back to CPU (Replit)
- Delivery/deploy branch is `colab-gpu` (feature-complete app that Colab + Replit both run)
- Always include the Colab GPU access link + steps when discussing DFM smoothness/lag
