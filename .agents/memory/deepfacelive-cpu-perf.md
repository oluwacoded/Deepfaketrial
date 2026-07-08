---
name: DeepFaceLive CPU performance tuning
description: Why live FPS is capped on the ~2-core CPU and which levers actually help.
---

# What caps live FPS
- Inference is server-side (onnxruntime, CPU) on ~2 cores. Live FPS ~= 1 / (server
  frame time + mobile round-trip).
- **DFM vs paste is a ~12x server-time split (measured, 320px frame):** DFM mode
  ~369ms/frame (the neural-net `convert()` alone is ~325ms and is irreducible on
  this CPU — no thread/graph tuning, quant, or input-size trick makes a per-frame
  net real-time); paste mode ~29ms/frame (no net, just detect+warp+blend). So
  "smooth on CPU" = paste mode; DFM smoothness needs a GPU. detect@384 ~26-43ms,
  decode/clahe/encode ~1ms each — negligible next to the net.
- **Browser GPU advice does NOT apply.** WebGL / WebGPU / TensorFlow.js "use the
  GPU" tips are irrelevant — no ML runs in the browser; it only captures/sends
  frames and draws the returned JPEG. Don't waste effort there.

# Levers that actually help (in order)
- **Adaptive detector `fixed_window`.** The face detector's `fixed_window`
  dominates per-frame detection cost. Scale it with the frame's longest side
  instead of hardcoding 640: small live frames (~320px) detect fine at 384 and
  run much faster (~0.75-1.0s -> ~0.4s measured); keep up to 640 for large
  uploaded photos so small faces aren't missed. run() is shared by
  live/photo/video, so this MUST stay size-adaptive, not a live-only special case.
- **Client transport: binary + non-blocking + serial.** Frames captured <=320px /
  JPEG q~0.55, sent ONE-IN-FLIGHT (serial gating — next frame only after the prev
  result). Transport is raw BINARY both ways over Socket.IO (client `toBlob` ->
  `arrayBuffer` -> emit; server accepts bytes | legacy {image:b64}, emits result
  bytes; client decodes via `createImageBitmap`) -> drops the ~33% base64 tax AND
  the synchronous `toDataURL` main-thread jank. KEEP the base64 + object-URL
  FALLBACKS (old iOS Safari) and arm the inFlight watchdog UP FRONT (before the
  async encode) — a failed/never-firing encode callback otherwise wedges the loop
  forever. Live result payload is tiny (~9KB) so live lag is LATENCY/compute-bound,
  not bandwidth — don't chase payload size.
- Lowering DFM `face_output_size` (224) would speed convert but hurts quality —
  off the table whenever the user wants *better* quality.

# Honest expectation-setting
- On ~2 cores, DFM live is ~2-3 FPS at best (frozen last-swap frame lagging head
  motion is inherent, not a bug); paste mode can be smooth. Don't promise smooth
  DFM video without a GPU.

# Gotchas
- Flask serves templates from an in-memory Jinja cache (auto_reload off): editing
  templates/index.html does NOT take effect until the workflow is RESTARTED. A
  screenshot/curl right after a template edit shows stale HTML otherwise.
