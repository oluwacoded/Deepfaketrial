---
name: DeepFaceLive Colab GPU deployment
description: The smooth path for the CPU-bound web face-swap = run it on a free Colab GPU; why delivery uses a slim colab-gpu branch, not master; device fallback + push quirks.
---

# DeepFaceLive on Colab GPU (the smooth path)

The web face-swap is CPU-bound on Replit (DFM `convert()` ~325ms → laggy). To make it
smooth, run the whole app on Google Colab's free T4 GPU (convert ~15ms). The phone→server
network hop stays; the GPU removes the compute bottleneck.

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
Notebook cells: set Runtime→T4 GPU, run 3 cells; the last prints a cloudflared
`https://...trycloudflare.com` link to open on the phone (HTTPS is required for camera
`getUserMedia`). Free Colab GPU disconnects after idle → just rerun the cells.

## onnxruntime-gpu on Colab can't find cuDNN (the "red cell" failure)
Unpinned `pip install onnxruntime-gpu` grabs the latest (1.22+), which needs CUDA 12 +
cuDNN 9 and expects those libs on `LD_LIBRARY_PATH`. On Colab the CUDA provider then fails
with `libcudnn.so.9: cannot open shared object file` (microsoft/onnxruntime #25609) — so it
silently drops to CPU, or a bare `import onnxruntime` check turns the notebook cell red.
**Fix that works:** also install the CUDA-12 nvidia wheels (`nvidia-cudnn-cu12`,
`nvidia-cublas-cu12`, `nvidia-cufft-cu12`, `nvidia-curand-cu12`, `nvidia-cusparse-cu12`,
`nvidia-cuda-runtime-cu12`, `nvidia-cuda-nvrtc-cu12`, `nvidia-nvjitlink-cu12`), then prepend
every `site-packages/nvidia/*/lib` dir to `LD_LIBRARY_PATH` **in the env dict passed to the
server subprocess** — setting it in the already-running notebook kernel is too late for libs
it has loaded; a fresh subprocess is what picks it up.
**Also:** never let the notebook's GPU check hard-crash — run `import onnxruntime` in a
throwaway subprocess wrapped in try/except and just print status. Combined with the code's
per-component CPU fallback, the app then runs even if the GPU never engages, instead of
showing a red error.

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
