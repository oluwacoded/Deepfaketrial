---
name: DeepFaceLive Colab GPU deployment
description: The smooth path for the CPU-bound web face-swap = run it on a free Colab GPU; why delivery uses a slim colab-gpu branch, not master; device fallback + push quirks.
---

# DeepFaceLive on Colab GPU (the smooth path)

The web face-swap is CPU-bound on Replit (DFM `convert()` ~325ms → laggy). To make it
smooth, run the whole app on Google Colab's free T4 GPU (convert ~15ms). The phone→server
network hop stays; the GPU removes the compute bottleneck.

## "Still looks like a clown / lags like mad" — triage the URL FIRST
Before re-debugging the swap engine, confirm the user is on the COLAB (GPU) link, not the
Replit (CPU) site — the SAME code runs on both and only Colab is smooth. The engine (GPU
auto-detect + the swap-quality/clown fix) can already be correct AND pushed; the usual real
gap is the user testing the CPU host, or their Colab cloning stale pre-push code. Fix: push
the current `colab-gpu` so the one-click link serves latest, then have them run the Colab
link (not Replit). Only dig into pipeline/quality after confirming they're actually on GPU
(get_providers() shows CUDA), not silently on CPU.

## Device auto-detect + graceful fallback (both components)
- The pipeline picks the device at runtime: prefers a non-CPU ORT device (CUDA) when
  present, honors `FORCE_CPU=1`, else CPU. Harmless on Replit (stays CPU).
- BOTH the detector AND the DFM model loader must fall back to CPU **independently**.
  **Why:** onnxruntime-gpu vs Colab CUDA/cuDNN mismatch is the #1 first-run failure. If
  only the detector falls back, a GPU model-init failure still leaves swap silently broken.
  Each component retries once on CPU when a GPU session raises.

## Delivery = slim `colab-gpu` branch, NOT master
- Colab clones the repo from public GitHub. **Never push master:** its `.git` history is
  ~3.6GB (bundles unused model data — InsightFaceSwap/LIA/S3FD split into `*.part*` chunks).
  GitHub rejects the push (PUSH_REJECTED, size).
- The web app only needs: all `.py` source + `templates/` + the YoloV5Face detector
  (single 28MB `YoloV5Face.onnx`, NOT split) + the tiny `modelhub/DFLive` catalog. DFM
  celebrity models download at runtime; the other face-swap models are never used here.
- `colab-gpu` is an **orphan** branch (no history) with only that slim set (~50MB, ~262
  files, zero `*.part*`). It pushes fine and is self-sufficient — verify with
  `git archive colab-gpu | tar -x -C /tmp/x` then import + init in `/tmp/x`.
- The repo `.gitignore` is a whitelist (`*` then `!*.py !*.md !*.txt !*.jpg !*.png
  !requirements*`), so `.onnx`/`.html`/`.ipynb` need `git add -f`.

## Git push quirks (bit me twice)
- `gitPush` publishes the **currently checked-out** branch. To push `colab-gpu` you must
  `git checkout colab-gpu` first; pushing while on master errors "current branch already
  tracks origin/master; cannot publish colab-gpu". The branch arg alone doesn't switch it.
- Switching master↔colab-gpu is safe for tracked files (git restores master's ~440 extra
  files from objects). Use `git checkout -f` to get past leftover untracked files
  (README.md etc.). To edit colab-gpu without disturbing the running master workspace, use
  a `git worktree`.

## One-click user delivery
Colab-opens-from-GitHub URL (works on phone or desktop):
`https://colab.research.google.com/github/oluwacoded/Deepfaketrial/blob/colab-gpu/deepfacelive_colab.ipynb`
Notebook is ONE form-mode cell (`#@title ... { display-mode: "form" }` hides the code and
shows just a play button — right for non-technical buyers): set Runtime→T4 GPU, tap it once.
It clones + installs + starts the server and prints a cloudflared `https://...trycloudflare.com`
link + QR to open on the phone (HTTPS is required for camera `getUserMedia`). Free Colab GPU
disconnects after idle → just tap the cell again. `raise SystemExit` is the clean way to stop
the cell (no scary traceback) on the no-GPU / install-failed paths.

## onnxruntime-gpu on Colab can't find cuDNN (the "red cell" failure)
Unpinned `pip install onnxruntime-gpu` grabs the latest (1.22+), which needs CUDA 12 + cuDNN 9.
The wheel does NOT reliably pick up pip-installed CUDA libs even when they are on
`LD_LIBRARY_PATH`, so `import onnxruntime` itself crashes with `libcudnn.so.9: cannot open
shared object file` (microsoft/onnxruntime #23643, #25609). The import raises
`import_capi_exception`, which kills the whole server — onnxruntime IS the inference engine, so
there is no "drop to CPU": if the import dies, nothing runs.
**Best fix (current):** `pip install "onnxruntime-gpu[cuda,cudnn]"` — the extra pulls the EXACT
matching CUDA + cuDNN nvidia wheels, so onnxruntime is self-contained and independent of
whatever CUDA/cuDNN Colab ships that week. Then call `onnxruntime.preload_dlls()` (GPU build
>=1.21 only) BEFORE any InferenceSession — it ctypes-loads those libs from site-packages,
bypassing the LD_LIBRARY_PATH/RPATH problem. The app does this at the top of `web_server.py`,
guarded by `hasattr(_ort, "preload_dlls")` so it is a harmless no-op on the CPU-only Replit
host (verified: prints "preloaded" then runs CPU inference fine).
**Belt & braces:** the notebook still prepends every `site-packages/nvidia/*/lib` dir to
`LD_LIBRARY_PATH` in the env dict passed to the server subprocess (a fresh process is what
picks up lib-path changes; the already-running kernel is too late).
**Why not hand-pin the nvidia-*-cu12 wheels (the old fix):** it works but drifts out of sync
with the onnxruntime build's exact cuDNN version; the `[cuda,cudnn]` extra lets pip resolve the
right versions instead.

## Branch feature-parity: `deploy-clean` holds the complete backend
Divergent branches carry DIFFERENT feature sets. `deploy-clean` is the FEATURE-COMPLETE
backend: `/api/target_face` paste-face mode, non-blocking inference (`eventlet.tpool` +
pipeline `snapshot()`/`run()`), binary WebSocket frames, WebRTC call rooms, `/api/process_video`,
and `static/favicon.ico`. `colab-gpu` originally added ONLY GPU auto-detect + CPU fallback and
was a REGRESSION missing all of that.
**Why it bites:** `templates/index.html` is byte-identical across branches, but its `fetch()`
calls hit endpoints (e.g. `/api/target_face`) that only some branches implement — so a branch
serves the full UI while its buttons 404 silently.
**How to apply:** the canonical running/pushed branch (`colab-gpu` — used by both the Colab
notebook and the Replit workflow) must carry BOTH deploy-clean's features AND the GPU fallback.
When switching or merging branches, diff the frontend's `fetch()` paths against the backend
routes (`grep -nE "@app.route|@socketio.on"`) to confirm parity before shipping.

## "On the GPU but laggy" = ONNX Runtime silently ran on CPU
onnxruntime can register a session on CPU even when you asked for a GPU EP: if the CUDA
EP fails to init it silently drops CUDA and keeps CPU, with NO exception. If the app
infers the device from HOST capability ("a GPU exists → report gpu"), the status LIES —
it claims GPU while inference crawls on CPU. That is the usual cause of "I'm on GPU but
it lags like a fool".
**Truth source:** `sess.get_providers()` on the ACTUAL InferenceSession — check for
`CUDAExecutionProvider`/`DmlExecutionProvider`. Record it per session (detector AND
model separately) and report THAT in status, not the host guess.
**Make the reason visible:** the code sets `sess_options.log_severity_level = 4`
(fatal-only), which HIDES onnxruntime's "falling back to CPU" warning. For GPU EPs lower
it to 2 (warning) so the cause (CUDA/cuDNN mismatch, missing lib) shows in the Colab
console; keep CPU sessions quiet at 4.
**Why:** it's a debugging trap — the swap still "works" and the banner says GPU, so you
chase the wrong thing. Trust get_providers(), not intent.
