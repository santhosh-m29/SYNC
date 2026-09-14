# SYNC local engine API

The frontend talks to the Python process through a small HTTP boundary. The
default loopback server is `http://127.0.0.1:8765`; set `VITE_ENGINE_URL` for a
different local address during development. Vercel only serves the static React
bundle and never runs this API.

`GET /api/state` returns `{ ready, status, error, state, audible }`. `state`
contains the current track, producer position, play state, queue order, Auto DJ
state, volume, current transition, cues, transition points, planned tracks,
transition status, vocal state, and bounded telemetry. `audible` is the output
clock snapshot used to keep the UI playhead aligned with the native audio buffer.

`GET /api/library` returns analyzed tracks with IDs, filenames, durations, BPM,
keys, and vocal-analysis metadata. `GET /api/waveform/{track_id}` returns a
bounded cached peak envelope; it may return `202 {"pending": true}` while a worker
decodes the source file.

`POST /api/control` accepts JSON commands: `play`, `pause`, `stop`, `previous`,
`next` with an optional track ID, `seek` with seconds, `cue` with track and
seconds, `point` with track and seconds, `queue` with a unique track list,
`volume`, `auto`, `transition`, and `clear_overrides`. The server validates these
commands before forwarding them to `PlaybackEngine`; the frontend does not import
or depend on Python classes.

`POST /api/upload` accepts an MP3, WAV, or FLAC body with an `X-Filename` header.
The local process writes it into `music/`, analyzes it in a worker, and registers
the resulting track. Existing files are never overwritten. The API is Host-bound
to loopback and permits browser CORS reads for the local engine endpoint so a
separately deployed frontend can connect when the user has explicitly started it.
