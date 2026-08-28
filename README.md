# AI DJ (SYNC)

SYNC is a local-first foundation for a future general-purpose AI DJ system. The
current milestone performs deterministic audio analysis only: it scans a music
directory, loads individual tracks, estimates tempo, beats/downbeats, Western
key, level-based energy, and compact spectral descriptors, then stores a
versioned JSON analysis cache. It does not yet recommend songs, train models, or
render transitions.

## Installation

Use Python 3.11 or newer and install the package with its development tools:

```bash
python -m pip install -e ".[dev]"
```

MP3 decoding depends on the local audio backend. Installing FFmpeg is recommended
for broadest codec support, although WAV and FLAC are supported through
SoundFile where the platform backend supports them.

## Analyze a library

```bash
python -m ai_dj analyze ./music
```

Supported input formats are MP3, WAV, and FLAC. Reusable cache entries are written
to `./music/.ai_dj_cache`, while plain `TrackAnalysis` JSON documents are written
to `./music/.ai_dj_analysis`; use `--cache-dir` and `--output-dir` to choose other
locations. Unchanged files reuse their cached analysis. Failed files are reported
while the rest of the batch continues.

## Current limitations

All musical descriptors are DSP estimates, not musical ground truth. Tempo
octave ambiguity remains possible. Downbeats are inferred only from common 3/4
and 4/4 accent patterns and expose confidence; the key estimate is a preliminary
Western chroma-template feature, and energy is a level proxy rather than a
perceptual-loudness model. Structure, recommendations, ML, and audio rendering
remain intentionally outside the current milestone.
