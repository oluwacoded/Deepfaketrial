---
name: eventlet greenlet I/O interleaving
description: When eventlet green-patches sockets/subprocess, where greenlets yield and why shared-state ownership must be claimed before the first yielding I/O.
---

# Eventlet greenlet I/O interleaving

Under eventlet `monkey_patch()`:

- **Socket I/O and `subprocess` are green.** `subprocess.run()` inside a greenlet
  cooperates with the event loop (the child runs in its own OS process, the wait
  yields instead of blocking) — safe to call ffmpeg etc. directly in a request/job
  greenlet without tpool.
- **Regular file I/O is NOT patched**, BUT a large upload `file.save()` (werkzeug)
  reads from the *green request socket* in chunks, so it **yields mid-write**.
  Other greenlets — a periodic janitor, another request — can run before the save
  returns.

**Why this matters:** a temp-video cleanup routine raced with in-flight uploads
because the job was registered *after* `file.save()`. During the mid-save yield,
the janitor greenlet saw the half-written upload as an orphan and nearly evicted
it under size-cap pressure.

**How to apply:** when adding background/janitor greenlets or any shared mutable
state touched across greenlets, assume interleaving at *every* socket read/write
and subprocess wait. Claim ownership first — register the job / mark the resource
active / take the lock — *before* the first yielding I/O (the save, the socket
read), never after. Treat "recently created, not yet registered" files as
protected too.
