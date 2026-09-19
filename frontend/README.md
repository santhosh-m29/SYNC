# SYNC browser DJ

Deploy this directory as a static Vite website. No Python server, access key,
account, database, cloud storage or API URL is required.

Visitors choose MP3, WAV or FLAC files with Add Songs. The browser reads selected
File objects, computes waveform/tempo/energy/section features in a Web Worker,
and mixes two Web Audio decks. Nothing is uploaded, persisted in browser storage,
or stored on a server. Refreshing/closing the page clears the session; original
files on the device are unchanged. Browser codec support determines accepted files.

## Deploy on Vercel

Set Root Directory to frontend. Run npm ci then npm run build; output is dist.
The root vercel.json also supports building from the repository root. Remove old
VITE_API_URL/VITE_ENGINE_URL variables; neither is used. Production HTML enforces
connect-src 'none', preventing upload/API/WebSocket connections. Web assets are
still downloaded normally. The app must first be loaded online; it is not a PWA.

## Analysis limits

The browser planner compares candidate songs and entry/exit phrase estimates using
onset tempo, energy, timbre proxy and repeat similarity. It avoids leading/trailing
silence and schedules sample-clock equal-power overlaps. This is a lightweight
heuristic, not verified verse/chorus or vocal recognition. It does not run Python
Demucs, harmonic key analysis or pitch-preserving tempo stretching. Unmatched beats
use short overlaps, preserving pitch. Manual cues/exits remain available.

Keep the tab active; device background suspension can stop browser audio. Imported
files are referenced in memory; decoded audio is bounded to current/future deck
buffers and a two-entry cache. No synchronization between devices is attempted.

Checks: npm run build, node local-analysis.test.mjs, node browser-engine.test.mjs.
