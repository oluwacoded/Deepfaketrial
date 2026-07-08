# DeepFaceLive Web

## Project overview
DeepFaceLive is a real-time AI face-swap application originally built as a Windows desktop app with a Qt GUI. This Replit fork runs it as a **web app** — your webcam stream is processed server-side with CPU-only ONNX Runtime and the swapped video is streamed back to the browser.

### Architecture
```
Browser (webcam) → WebSocket → Flask/SocketIO server → Pipeline → WebSocket → Browser (output)
```

Pipeline stages (all CPU, no GPU required):
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

### Performance
- CPU-only: expect ~1–3 FPS
- First model load: downloads ~100–200 MB .dfm file from GitHub releases
- Models are cached in `userdata/models/`

## User preferences
- Keep the original Qt desktop app code intact
- Web interface is the new entry point
- CPU-only inference (no CUDA/DirectX required)
