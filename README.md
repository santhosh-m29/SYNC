# AI DJ (SYNC)

SYNC is a local-first foundation for a future general-purpose AI DJ system. The
current milestone provides deterministic analysis and transition planning: it
scans a music directory, extracts rhythmic, harmonic, energy, spectral, and
structure candidates, then stores a versioned JSON analysis cache. It does not
yet train models or render transitions.

## Deterministic compatibility baseline

`ai_dj.matching.score_track_pair(track_a, track_b)` produces an explainable
candidate-transition score; `rank_next_tracks(current, candidates)` sorts those
results. This is a deterministic benchmark, not an ML model and it does not
render or modify audio.

The overall score is the confidence-weighted mean of these `[0, 1]` components:
tempo (0.25), harmony (0.20), rhythm (0.15), energy (0.15), structure (0.10),
vocal activity (0.05), and timbre (0.10). A component's effective weight is its
base weight multiplied by its analysis confidence. Unavailable evidence—for
example the current unimplemented local vocal detector—has zero effective
weight rather than being treated as certainty.

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
perceptual-loudness model. Structure candidates are non-semantic and need not
correspond to verse/chorus labels. The compatibility and transition systems are
transparent baselines, not final DJ-set planning. ML and audio rendering remain
intentionally outside the current milestone.

## Transition candidate planning

`ai_dj.transition.find_best_transitions(track_a, track_b)` returns ranked,
non-rendering `TransitionPlan` objects; `find_best_transition` returns the first
valid plan or `None`. Exit and entry positions are derived only from analyzed
phrases, sections, bars, and downbeats. The planner combines the Phase 3 tempo
and harmony scores with local beat, phrase, energy, structure, and (when
available) vocal scores. It selects only feasible 16-, 8-, or 4-second plans.
The strategy field is descriptive (`phrase_crossfade`, `outro_intro`,
`energy_rise`, `energy_drop`, or `instrumental_entry`) and does not cause audio
to be mixed or rendered.

## Transition datasets

`ai_dj.datasets.generate_dataset(tracks, output_dir, config)` creates a new,
versioned directory containing `train.jsonl`, `validation.jsonl`, `test.jsonl`,
and `metadata.json`. JSONL supports streaming large datasets while preserving
nested feature groups. Rows contain track IDs and transition features only—never
raw audio—and carry `dataset_version`, `analysis_version`, and `feature_version`.

Candidates are sampled deterministically across the full transition-score range,
so weak-label datasets retain more than the top plans. Labels are automatic weak
supervision (`label_source: automatic`), not human quality ground truth. Supply
artist/album/source groups through `DatasetConfig.group_by_track_id` to prevent
those groups from crossing train/validation/test splits. Human evaluation forms
can be exported/imported with the annotation helpers; imported human or
consensus labels are returned as new examples and never overwrite the automatic
artifact in place.
