# SYNC workstation

From the project directory on Windows:

```powershell
.\.venv\Scripts\python.exe -m ai_dj ui
```

Open **http://127.0.0.1:8765**. The server binds only to this computer. Audio uses
the existing native output device, not browser audio. The initial library analysis
and ten-track plan finish before playback is enabled. Existing caches are reused.

Optional arguments: `--port 8765`, `--device DEVICE_ID`, `--sample-rate 48000`,
and `--no-vocal-analysis`. Vocal estimates appear only when the engine has them;
the thin strip beneath the cue waveform represents estimated vocal activity.

## Cue and playback workflow

- Select a library row to inspect its waveform without changing playback.
- Click the waveform, use left/right arrows for 0.1-second changes, or type exact
  cue seconds and press **Set cue**.
- **Load track** loads that cue on the deck. While stopped, press **Play** to begin.
  While playing, Load/Skip performs a live handoff.
- Play/Pause, Previous, Next, Stop, volume and the seek bar control PlaybackEngine.
- **Auto DJ** enables automatic continuation. With it off, the current track
  stops at its end unless a manual transition has been scheduled. **Mix now**
  initiates an immediate live handoff.

## Queue and timeline

The initial queue contains the library; its displayed order follows the planned
chain. Dragging/removing tracks makes queue order explicit. A track occurs once
per queue; adding an already queued track moves it. The playing track stays in
the queue. The queue cycles as in the existing live playback system.

Drag **EXIT** to change an outgoing media timestamp and **CUE** to change the
incoming media timestamp. Drag an incoming track's title horizontally to move its
start relative to its predecessor. Drop a library or queue track onto a timeline
lane to insert it after that lane at the dropped time. Editing the playing lane
requires a transition at least half a second ahead of the producer position.

The timeline shows the current deck and up to two planned incoming decks; the
queue shows the rest. Its ruler is time relative to the current deck's start,
adjusted for playback rate. Marker labels and numeric controls use source-file
seconds. **VIEW** changes the visible time span. The cue editor has independent zoom.
**Reset cues** clears cue and exit overrides. Manual queue order remains explicit.

Manual edits invalidate stale preparation and show **MANUAL** once scheduled.
During preparation the status says so; waveform rendering never creates audio.
Playheads use the engine's output-frame history and device latency, with bounded
interpolation between state updates. They freeze on a disconnected page.

## Adding music

Use **+ Add Music** or drop MP3, WAV or FLAC files onto the page. Files are copied
into `music`, analyzed in a separate process and registered with the running
engine. Imports never overwrite an existing filename. The per-file limit is
250 MB. Imported tracks appear in Library; add them to Queue when ready.

Waveform envelopes are computed from decoded audio in a separate process, bounded
to 12,000 peaks, and cached in `music/.ai_dj_waveforms` using file size and modification
time. PCM playback and analysis retain their existing caches. No external web
assets, frontend build step or additional web framework is required.

## Validation

`tests/test_ui.py` exercises cue loading, manual scheduling with Auto DJ disabled,
queue removal/reloading, volume, natural end, pending-load playback, real peak
generation/cache reuse, HTTP controls and imports.

The real-device check in `output/ui-live-validation.json` used AMORE, CALIENTE!
and MALÍCIA from the local music folder. It exercised cue playback, pause/resume,
an automatic transition, skip, seek and a manual transition with Auto DJ off:
three completed handoffs, zero ring underruns, zero device underflows and no
engine errors. Optional vocal separation was disabled for this run; it does not
establish vocal-matching quality or guarantee silence-free source material.
