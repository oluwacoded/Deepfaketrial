---
name: DeepFaceLive env recovery & branch hazard
description: How to recover when the project loses its .replit (no python, no workflow, "failed project"), and why branch-switching this repo is dangerous.
---

# "Failed project" = lost `.replit`
Symptoms: `python`/`python3` not found on PATH, the app workflow is gone, preview
is blank, and `.cache/replit/toolchain.json` shows the placeholder "Please use
the Configuration pane to configure your run command."

**The interpreter and deps usually still exist** — the Nix store python and the
installed packages under `.pythonlibs` survive; only the `.replit` provisioning
(modules + run config + workflow) was removed.

**Recovery (no code changes needed):**
1. Reinstall the language module via package-management (`installProgrammingLanguage`,
   here `python-3.12` — matches the `.pythonlibs` wrapped interpreter). This puts
   `python` back on PATH and regenerates `.replit`.
2. Reconfigure the workflow: webview, port 5000, `python web_server.py`.
3. Verify: `/` 200, `/api/models` returns the DFM list, WebSocket shows Connected.

**Why:** losing `.replit` looks catastrophic but is purely a config loss; a module
reinstall + workflow reconfigure fully restores it without git surgery or edits.

# Branch hazard: main vs master track DIFFERENT file sets
- `main` (full-featured: 575-line web_server with WebRTC /call + Colab notebook)
  tracks only ~28 files — the web app source (xlib/, modelhub/, templates/,
  YoloV5Face.onnx) is UNTRACKED on disk (force-added / gitignored).
- `master` (clean refactor: 142-line web_server, NO /call backend, no Colab)
  tracks ~700 files — all source is committed.
- **Therefore `git checkout` between them can DELETE the untracked source from
  disk.** When only asked to "get it working," prefer recreating run config on
  the CURRENT branch over switching branches.
- Note: the UI (index.html) on master still shows a "Start Video Call" button,
  but master's backend has no `/call` route, so that feature is dead on master;
  the basic swap (webcam frames, photo upload, DFM models) works.

# "Nothing to preview" = reset silently put you back on `master` + dropped the workflow
Symptom (often reported from the phone): the preview pane shows "No previewable Apps
yet". Cause: a checkpoint/restore left the working tree on `master` — the OLD stripped
app (tiny `web_server.py`, no `static/`, no `/api/target_face`) — and removed the run
workflow. The app is NOT broken; it's the wrong branch with no process running.
**Recover:** `git fetch origin colab-gpu` → `git checkout -f colab-gpu` →
`git reset --hard origin/colab-gpu` (the complete app is committed there and on GitHub),
then ensure a workflow runs `python web_server.py` on port 5000 (webview) — it often
re-adds itself on the branch switch. Verify `/` and `/static/favicon.ico` return 200.
**Why:** `colab-gpu` is the canonical app but an orphan branch; the platform's default
branch (`master`, the regression) can reassert itself on restore. ALWAYS confirm the
current branch before treating a "broken app" report as a code bug.
