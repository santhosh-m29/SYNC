# AI DJ (SYNC)

SYNC is a local-first foundation for a future general-purpose AI DJ system. The
current milestone provides deterministic analysis, planning, and plan-driven
transition rendering: it
scans a music directory, extracts rhythmic, harmonic, energy, spectral, and
structure candidates, then stores a versioned JSON analysis cache. It does not
does not yet provide a validated learned DJ-ranking model.

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

## Generate an autonomous set preview

```bash
python -m ai_dj generate --input ./music --output ./output/set.wav \
  --tracks 4 --trajectory maintain
```

Generation reuses the analysis cache, selects a deterministic start track,
cheaply prefilters a bounded candidate pool, runs the context-aware beam planner,
renders eligible transitions, and assembles their WAV previews. It writes
`set.report.json` for machine consumption and `set.report.txt` for a concise
human-readable sequence and transition summary. `--seed`, candidate-pool size,
beam width, trajectory, and target track count are recorded in the report.

The generator remains usable without an ML artifact and currently reports that
deterministic fallback explicitly. A learned model is not used until it is
validated on a non-empty, leakage-safe, human/consensus-labeled holdout.

## Current limitations

All musical descriptors are DSP estimates, not musical ground truth. Tempo
octave ambiguity remains possible. Downbeats are inferred only from common 3/4
and 4/4 accent patterns and expose confidence; the key estimate is a preliminary
Western chroma-template feature, and energy is a level proxy rather than a
perceptual-loudness model. Structure candidates are non-semantic and need not
correspond to verse/chorus labels. The compatibility and transition systems are
transparent baselines. The renderer is an initial preview engine, not a
production mixer; ML remains an experiment until a valid held-out human-labeled
dataset exists.

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

## ML ranking and evaluation

`ai_dj.models.train_transition_quality_model(dataset_dir, artifact_dir)` trains
a `HistGradientBoostingRegressor` for the continuous `[0, 1]`
`transition_quality` target using the predefined dataset splits—never a random
reshuffle. It writes a non-overwriting versioned artifact with `model.pkl`,
`metrics.json`, and `MODEL_CARD.md`; `predict_transition_quality` uses the same
feature preparation at inference.

The evaluator reports MAE, RMSE, and Pearson correlation for the learned model
and deterministic plan-score baseline. Automatic labels are generated from that
same plan score, so the deterministic baseline has zero error by construction;
such data cannot demonstrate ML improvement. Meaningful comparisons require
human/consensus labels on diverse, group-safe held-out tracks.

See [the model card](docs/TRANSITION_QUALITY_MODEL_CARD.md) for intended use,
evaluation requirements, and limitations.

`ai_dj.evaluation.rank_transition_candidates(current, candidates, model)` keeps
three independent rankings: deterministic baseline, learned prediction, and a
hybrid. The hybrid formula is `0.8 × ML prediction + 0.2 × plan confidence`;
it excludes invalid timestamps/durations and tempo corrections above 12% after
accounting for 1:1 and half/double-time relationships. These are technical
guardrails, not a claim that all musically unusual transitions are bad.

`evaluate_ranking_systems(dataset_dir, model)` reports held-out regression and
ranking metrics for all three systems plus transparent diagnostic indicators.
It rejects an empty test split. `export_blind_comparisons` creates evaluator
manifests with randomized opaque options and a separate answer key; it does not
render audio, so listening requires an external playback workflow at this stage.

## Context-aware set planning

`ai_dj.set_planning.plan_set(start_track, candidates, config)` uses a bounded,
deterministic beam search to return a JSON-serializable `SetPlan`; use
`plan_set_greedy` and `compare_greedy_and_sequence_aware` to compare it with a
one-step baseline. It never repeats a track by default, excludes technically
ineligible transitions, and can guide a `build`, `maintain`, `release`, or
`peak` energy trajectory. Optional caller-supplied `artist_by_track_id` values
penalize recent artist repetition, while tempo/key/timbre similarity supplies a
light diversity penalty.

The objective is intentionally fixed and inspectable: 65% eligible transition
score, 20% energy-trajectory fit, and 15% variety. This is a deterministic
search baseline—not a trained model.

## Transition rendering

`ai_dj.rendering.render_transition(source, destination, plan, output_path)`
executes an existing, eligible `TransitionPlan`; it does not choose tracks or
timestamps. It creates a float WAV preview consisting of source pre-roll,
plan-duration overlap, and destination post-roll. Supported rendering strategies
are `phrase_crossfade`, `instrumental_entry`, and `outro_intro`.

Destination audio is pitch-preserving time-stretched with librosa only when the
residual tempo correction is within ±12% (half/double-time relationships are
not stretched). Beat-grid offsets may make a bounded ±50 ms destination shift.
The renderer performs conservative RMS gain matching (at most ±6 dB) and peak
protection, and returns a `RenderResult` with sample rate, timing, gain, peak,
RMS, clipping count, stretch rate, and alignment offset. It does not yet apply
EQ, stems, source separation, or loudness-standard metering.

The assembled set is an ordered sequence of validated transition previews, not
a full-length mastered DJ mix. The report exposes this limitation along with
analysis failures, technical checks, fallbacks, and phase timings. Final set
assembly uses a conservative 20 ms equal-power join and records boundary-jump,
beat-offset, RMS-range, finite-sample, clipping, and unavailable-artifact
diagnostics. See [the system audit](docs/SYSTEM_AUDIT.md) for measured baseline
and improvement evidence, including a 32-track local-library integration run.

## Vocal-aware planning

Normal playback is fail-closed: it requires trusted vocal timelines for both
tracks and rejects every sustained significant vocal-vocal overlap. The ML
ranker cannot override this deterministic gate. It favors vocal-safe phrase
and beat-aligned alternatives, records the incoming vocal entrance, and uses an
instrumental entry when the next vocal naturally starts later. Existing mixed
audio is intentionally marked as vocal activity unavailable; the system does
not fabricate vocal predictions from generic DSP and will not claim a
vocal-safe crossfade without validation. See
[vocal-aware transitions](docs/VOCAL_AWARE_TRANSITIONS.md) for the capability
boundary and evaluation status.
