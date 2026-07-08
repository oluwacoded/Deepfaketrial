---
name: DeepFaceLive DFM swap quality (the "melted clown" fix)
description: Why the celebrity swap looks soft/ghosted like a "clown" and how face_coverage interacts with the model's fixed low resolution; the levers that make it sharp.
---

# DFM swap quality: coverage vs the model's fixed resolution

DFM celebrity models have a FIXED, small native resolution (Jackie Chan = 224x224;
many are 224/256/320). That is a hard detail ceiling — you cannot get more facial
detail out than the model's res, no matter how big the input frame is.

## The "melted clown" root cause
`face_coverage` controls how WIDE the aligned crop is around the detected face
(higher coverage = wider crop = the real face fills LESS of the crop). That crop is
resized DOWN to the model input res. So with wide coverage the face occupies only a
fraction of an already-small input:
- coverage 2.0 → the real face is only ~half of the 224 crop → the model sees AND
  generates it at ~110px → upscaling that back onto a larger in-frame face gives a
  soft, melted, often GHOSTED (doubled features) result. That is the "clown" face.
- The ghosting is misalignment/scale, not color — color match was roughly fine.

## The fix that worked (validated by eye, not guessed)
- Tighten `face_coverage` to ~1.4–1.6 so the real face fills most of the model input
  → the model sees & generates far more detail AND aligns better. This is the single
  biggest quality lever, and it fixes both softness and ghosting.
- Cut/merge at an output size ABOVE the model res (e.g. 320) and warp the swap up with
  LANCZOS4 (not LINEAR) for a cleaner upscale.
- Add a light unsharp mask on the warped swap (amount ~0.5–0.6, clip back to [0,1])
  to counter the residual softness from the low model res.
- **Why not just raise output_size:** it cannot add detail past the model's native
  res; coverage is what decides how many real-face pixels reach the model.

## How to validate (do this — don't guess-and-flail)
Reproduce on CPU with `FORCE_CPU=1`: load the real `.dfm`, run the REAL pipeline on a
handful of downloaded faces, and SAVE labelled before/after montages to look at. Sweep
coverage/interp/sharpen and pick by eye. Close-up test photos exaggerate blur (face >>
224px); a phone selfie (face a few-hundred px) looks much better. The live client sends
only ~320px frames, so the in-frame face is already near model res there — meaning
coverage/ghosting, not resolution, is the dominant lever at real usage size.
