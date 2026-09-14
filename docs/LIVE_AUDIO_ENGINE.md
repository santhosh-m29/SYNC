# Live audio engine

## Architecture inspected

The existing code is a Python DSP/CLI project; it has no UI or playback device
layer. `ingestion/scanner.py` discovers files and `ingestion/loader.py` decodes
mono analysis audio. `pipeline/analyze.py` runs tempo, beats, inferred downbeats,
key, energy, spectral and structural analysis. `pipeline/cache.py` fingerprints
source identity, size, modification time and analysis version. These remain in use.

`matching/compatibility.py` provides confidence-weighted deterministic ranking.
`transition/planner.py` searches structural entry/exit candidates, and
`transition/vocal_safety.py` checks source timelines with rate mapping.
`set_planning/planner.py` performs offline sequence search. Datasets, evaluation,
and the experimental transition-quality model are offline evaluation tools;
they are not a validated vocal model or a playback scheduler.

`rendering/renderer.py` and `pipeline/generate.py` assemble WAV previews. They
remain available for existing evaluation workflows, but the live engine does
not call either. `analysis/vocals.py` previously returned unavailable for every
track. `rendering/stems.py` already offered cached Demucs separation.

The new `playback` package separates these responsibilities:

- `audio.py`: stereo decoding, fixed gain/headroom, worker-side pitch-preserving
  preparation of individual incoming tracks.
- `planner.py`: compatibility selection and future events using the outgoing
  deck's effective tempo, media clock, vocal evidence and manual overrides.
- `queue.py`: order, current item, recent history, automatic/manual provenance,
  and no repeats until candidates have been visited.
- `mixer.py`: two decks, media/sample clocks, sample-exact equal-power fades,
  continuous incoming playhead, de-clicking and peak protection.
- `engine.py`: commands, generation-based cancellation, a separate process for
  Python-heavy planning and a worker thread for tempo preparation,
  deadline handling, lifecycle and immutable state snapshots.
- `device.py`: one PortAudio output stream, bounded single-producer/
  single-consumer PCM ring, callback copy only, underrun counters.
- `validate.py`: repeatable real-file and real-device validation.

## Running

Use Python 3.11 or newer (3.12 tested):

```powershell
python -m pip install -e '.[dev,playback,vocals]'
python -m ai_dj play music --vocal-stems
```

Demucs model weights must be available locally, or downloaded on the first
preparation run. Cached structural analysis and the first two stereo decks are
prepared before output starts. Vocal analysis is attempted by default in a
separate process, prioritizing current/upcoming tracks; `--no-vocal-analysis`
explicitly skips it. Completed vocal evidence is read immediately from cache.
New evidence updates future scheduling without disturbing an active fade or
manual transition. Playback never waits for a transition WAV. Incoming tempo
preparation and transition planning run while the current song plays.

The core API is independent of the CLI:

```python
from ai_dj.playback import PlaybackEngine

with PlaybackEngine.from_library("music", vocal_stems=True) as engine:
    engine.play()
    engine.start_output()
    # Keep this scope alive for playback. Commands may come from another thread.
    engine.pause()
    engine.play()
    engine.seek(42.5)
    engine.set_cue(next_track_id, 8.0)
    engine.set_transition_point(current_track_id, 180.0)
    engine.reorder_queue(track_ids)
    engine.transition_now(duration=2.0, track_id=next_track_id)
    engine.next()
    engine.previous()
    engine.stop()
    snapshot = engine.state
```

On Windows, put application startup in `if __name__ == "__main__":` so the
planning process can start safely. The provided CLI already does this.

Controls are asynchronous. `state.position` is the producer's media position;
audibility trails by `buffered_seconds` plus device latency. Do not call
`process()` concurrently with `start_output()`; direct processing is intended
only for deterministic headless tests. Cues and exit points persist until
`clear_overrides()`. A point behind a seek is retained but cannot fire in the
past. Skip uses a short live fade into the resident target at its cue, then
plans that track's next event. No extra playback stream is launched.

## Vocal evidence and musical limits

Demucs stem activity uses short RMS frames, absolute and relative floors,
hysteresis, attack context and release. It operates on the separated vocal
source, not a mid-frequency proxy in a mixed waveform. Cached activity is
versioned alongside fingerprinted stems. Its activity indicators are **not
calibrated probabilities**: method and confidence explicitly identify this,
and confidence remains 0.5. Failed/missing separation reports unavailable.
Trusted timelines provided to the engine can certify a no-collision window;
uncertain stem evidence can rank windows but cannot certify them.

The live planner searches the existing structural candidates at multiple
durations, avoids automatic cues inside detected vocal segments, and maps
both timelines through the actual playback rates. Reliable bar evidence
quantizes duration. Large/uncertain tempo corrections use native rate and a
short fade. Half/double-time relationships are considered with an 8% limit.
The incoming track retains its matched rate after promotion, avoiding a tempo
jump. Later events use that effective tempo rather than the file's original BPM.

Before `play()` starts, the engine builds a ten-track lookahead chain from the
current queue. Each source gets a destination, outgoing point, incoming cue,
duration, tempo ratio and safety state. Only the next event is attached to the
mixer at a time, but later decisions already exist, so playback does not wait
until a song is ending to decide what follows. After five songs complete, the
next ten-track window is rebuilt in the preparation worker while the active
window continues. Seek, queue reorder, cue changes and transition overrides
invalidate and rebuild the chain.

When evidence cannot establish a safe overlap, the engine uses a short smooth
handoff and reports unverified. It does not promise zero vocal overlap for
unknown material. Source separation itself is not calibrated singing detection;
labelled vocal timelines and listening evaluation are still needed before
claiming reliable vocal safety on an arbitrary library. Existing downbeats
and four-bar phrase groups are also confidence-scored heuristics.

## Buffering and operational limits

The engine keeps at most three native stereo tracks in its decode cache, plus
the current deck and tempo preparation references. It enforces a per-track
memory admission limit (default budget 1 GiB, with reserve for variants).
Incoming songs are decoded and stretched off-thread, well before their event.
The library itself is metadata: its total duration is not limited by resident
PCM memory. Individual exceptionally long tracks can exceed the preparation
budget; increase it explicitly or split those files. DSP/model working memory
is additional to the PCM budget.

If tempo preparation misses its deadline, native-rate audio is available for
a brief unverified handoff. If an entire decode is late, the engine uses another
resident track and preserves a pending manual request. A cold manual skip keeps
the current song playing until the requested target is ready, then fades into
it. It never waits in the producer or callback. Missed plans cannot
be installed after their media timestamp. Device callback code does no analysis,
DSP, file I/O, logging, queue-lock acquisition or waits. Python/NumPy/PortAudio
on a general-purpose OS is **soft real time**, not a hard guarantee that an OS
stall can never underrun. Underruns and device status events are observable.

## Validation

```powershell
python -m pytest -q
python -m ai_dj.playback.validate music --seconds 1800 --device-seconds 60 --sample-rate 48000 --device 8
python -m ai_dj.playback.validate music --seconds 1800 --vocal-stems --report output/vocal-live-validation.json
```

Device IDs and supported sample rates are machine-specific; omit `--device`
to use the default output. This machine's MME default supports 44.1 kHz; its
WASAPI endpoint requires 48 kHz. Use `play --sample-rate 48000 --device 8`
when choosing that endpoint explicitly.
The harness processes real decoded music through the same live mixer, checks
finite audio, peak limits, silent transition blocks, scheduling, cues, queue,
seek, pause, skip and previous, and optionally exercises the hardware stream.
The accelerated test driver waits between songs for preparation because a
simulated minute can elapse in less than a second; the live path never waits.
Separate regression tests deliberately leave preparation unfinished to exercise
the deadline fallback. Reports are JSON under `output`, not transition clips.
