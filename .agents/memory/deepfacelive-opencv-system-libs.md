---
name: DeepFaceLive OpenCV system libs
description: cv2 import breaks after an env reboot; restore the Nix libs.
---

# OpenCV (cv2) system libraries on this repl

- After a package install or environment reboot, `import cv2` can fail with
  `libxcb.so.1: cannot open shared object file` (or libGL / glib / X11). The
  Python opencv wheel is still installed, but the Nix system libraries it links
  are missing from the runtime linker path — the workflow then dies at import
  before Flask ever starts.
  **Why:** a language-package install reboots/rebuilds the Nix env and can drop
  previously-present system libs. opencv (esp. the non-headless build) links
  libxcb/libGL/glib, and its Qt platform plugin links the X11 set.
  **How to apply:** restore via `installSystemDependencies` with Nix names:
  `xorg.libxcb`, `libGL`, `glib`, and for the Qt plugin
  `xorg.libX11 xorg.libXext xorg.libSM xorg.libICE`. Diagnose the exact set with
  `ldd <path-to>/cv2/*.so | grep 'not found'`. These end up in `replit.nix` —
  commit it so the fix persists.
