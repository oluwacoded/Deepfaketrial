---
name: DeepFaceLive WebRTC call-room lifecycle
description: Why signaling rooms must survive a caller socket disconnect on mobile, and how ICE must be routed by role to survive caller SID churn on reconnect.
---

# WebRTC call-link rooms (mobile survival)

The share-a-call link is peer-to-peer WebRTC with the server as a signaling relay. Rooms are
an in-memory dict keyed by room id, holding the caller's offer, the callee's answer, and
buffered ICE candidates.

## Never destroy a room when the caller's socket disconnects
On a phone the caller's socket drops as a NORMAL part of the flow: the instant they leave the
browser tab to copy/share the link, lock the screen, or the tunnel hiccups, socket.io
disconnects. If disconnect deletes the room, the callee opens the link to a 404 "expired" —
the exact bug users report.
**Rule:** rooms live until their TTL, an explicit End Call, or the caller starting a new call.
On disconnect, only opportunistically sweep TTL-expired rooms; do NOT drop the disconnecting
caller's room.
**Recovery:** persist the callee's answer and buffer the callee's ICE on the room, and provide
a `rejoin_call` that re-points the room to the caller's NEW socket id and replays the stored
answer + buffered ICE. The client emits rejoin on socket reconnect while a call is active.
**Why:** without this, every real mobile share breaks; with it the caller can background the
app, share the link, and come back.

## Route ICE by role, not by "== caller_sid else callee"
On reconnect the caller gets a NEW socket id, and socket.io may flush its buffered ICE emits
BEFORE the rejoin handler updates caller_sid. A naive `sid == caller_sid ? caller : callee`
then files the caller's own candidates into the callee buffer — which is only flushed to the
callee, never back to the caller — so they're lost and negotiation can stall.
**Rule:** before the callee answers, the only side trickling ICE is the caller → treat all ICE
as caller ICE (buffer until the callee joins). After the answer, only candidates from the
KNOWN callee_sid are callee ICE; everything else (including the caller's new pre-rejoin sid) is
the caller, so caller candidates are never misfiled. Always relay live to the room too;
duplicate candidates are harmless because clients swallow addIceCandidate errors.

## Callee page must not permanently short-circuit
A duplicate-connect guard (`if (rtcConn || connecting) return`) must be paired with tearing
down rtcConn on socket disconnect, or a reconnect can never renegotiate and the user is stuck
until a manual refresh. Full mid-call bidirectional renegotiation (ICE restart on both peers)
is a bigger feature; the minimal safe fix is to reset rtcConn/connecting on disconnect so a
reconnect at least retries.
