---
name: DeepFaceLive publishing (Replit deploy)
description: Why the Replit publish build failed (workspace too big for the security scan) and why this app must be a Reserved VM, not autoscale.
---

# Publishing DeepFaceLive on Replit

## Build failed at the "Security Scan" step ("connection lost")
Replit deployments package ALL workspace files by default (8 GB cap for VM/autoscale;
`.git` is usually auto-excluded). This repo ballooned to ~6.8 GB and every publish died a
few seconds in at "Running Security Scan" -> "Security scan skipped: connection lost" ->
failed — i.e. the scanner choked on the VOLUME, not a code/build error. The 4-line build
log with no compile/run error is the tell.
**Fix:** keep the shipped payload lean with `.replitignore` (gitignore syntax, honored by
deploys). Exclude what is NOT needed to run: `userdata/` (DFM models download at runtime;
`MODELS_DIR.mkdir(exist_ok=True)` recreates the dir at import), `.cache/`, `__pycache__/`,
`attached_assets/`, `.git/`. That cut the payload from ~6.8 GB to ~560 MB.
**Why it recurs:** the 686 MB `Jackie_Chan.dfm` is a runtime download that keeps landing in
`userdata/`, and `.cache` regrows — so the ignore rules, not one-time deletes, are the fix.

## This app MUST be Reserved VM, never autoscale
The default publish target was autoscale (Cloud Run, stateless, spins down when idle) —
wrong here. It needs an always-on server: a Telegram bot on long-poll, live WebSocket frame
streaming (eventlet + socketio), an in-memory loaded face model, and in-memory WebRTC
signaling rooms. Autoscale kills the bot when idle and drops WebSocket state.
**How to apply:** set `deploymentTarget = "vm"` + `run = ["python","web_server.py"]` in
`.replit [deployment]`, and have the user confirm "Reserved VM" in the Publishing pane (the
pane is authoritative for an already-published deployment).
**Why it's the user's call:** VM is billed always-on vs autoscale's per-use — a real cost
difference, so state the trade-off and let them choose at Publish time.

## Caveat: one Telegram token, one poller
`TELEGRAM_BOT_TOKEN` can only be long-polled by ONE process. If the dev workflow AND the
published VM both run the bot, Telegram returns 409 Conflict. After publishing, only one
should poll (stop the dev bot, or let prod win).
