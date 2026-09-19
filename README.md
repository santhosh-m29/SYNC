# AI DJ (SYNC)

The `frontend/` is now a standalone browser DJ: select files from your own device,
analyze and mix locally with Web Audio. No uploads, accounts, persistent storage
or Python service are required for the website. See [frontend deployment and
analysis limits](frontend/README.md). The native Python engine below remains a
separate optional application.

SYNC is a local audio-analysis and live DJ playback engine. It scans a music
directory, caches rhythmic, harmonic, energy and structure analysis, and mixes
two decks through one buffered audio stream. Queue and playback controls are
available through a UI-independent API. Offline transition rendering remains
available for evaluation. Vocal evidence carries explicit uncertainty; there
is no validated learned DJ-ranking model.

Start playback with `python -m ai_dj play music`. See
[the live engine guide](docs/LIVE_AUDIO_ENGINE.md) for dependencies, controls,
hardware configuration and validation limits.

## Deterministic compatibility baseline

`ai_dj.matching.score_track_pair(track_a, track_b)` produces an explainable
candidate-transition score; `rank_next_tracks(current, candidates)` sorts those
results. This is a deterministic benchmark, not an ML model and it does not
render or modify audio.

The overall score is the confidence-weighted mean of these `[0, 1]` components:
tempo (0.25), harmony (0.20), rhythm (0.15), energy (0.15), structure (0.10),
vocal activity (0.05), and timbre (0.10). A component's effective weight is its
base weight multiplied by its analysis confidence. Unavailable evidence—such as
vocal activity from an ordinary mixed waveform—has zero effective weight rather
than being treated as certainty.

## Installation

Use Python 3.11 or newer and install the package with its development tools:

```bash
python -m pip install -e ".[dev]"
```

MP3 decoding depends on the local audio backend. Installing FFmpeg is recommended
for broadest codec support, although WAV and FLAC are supported through
SoundFile where the platform backend supports them.

## How the AI/DSP engine works

SYNC's production path is an explainable audio-analysis and scheduling system.
The live engine does not currently use a trained neural network to decide what a
DJ should play. It uses deterministic DSP features, confidence values, and
rule-based candidate search. The optional transition-quality model is an offline
experiment and is not allowed to override the live planner's safety checks.

For each MP3, WAV, or FLAC file, ingestion scans the library and decodes a mono
analysis signal with a consistent sample rate. The analyzer then builds one
versioned `TrackAnalysis` record containing:

- tempo: librosa onset strength and tempo estimation, normalized to a plausible
  40–240 BPM range while retaining half/double-time alternatives and periodicity
  confidence;
- beats: beat timestamps from the onset envelope, with confidence based on beat
  regularity and onset strength;
- downbeats and bars: a confidence-scored 3/4 or 4/4 accent-phase heuristic;
- energy: normalized RMS level plus a coarse one-second energy timeline;
- key: a preliminary chroma-template estimate using major/minor pitch profiles;
- timbre: spectral centroid, bandwidth, rolloff, spectral contrast, and MFCC
  summaries;
- structure: bar-aligned phrases, non-semantic section boundaries, section
  confidence, and repetition IDs from normalized chroma/MFCC/energy similarity;
- vocal activity: unavailable for ordinary mixed audio unless optional Demucs
  separation produces cached vocal and accompaniment stems.

These descriptors are estimates, not musical truth. Structure labels are kept as
`other`; the analyzer does not claim that a boundary is definitely a verse,
chorus, or bridge. Every feature that can be uncertain carries confidence so
missing or weak evidence reduces its influence rather than becoming a fabricated
prediction.

Analysis is cached using the source identity, file size, modification time, and
analysis version. A repeated scan reuses valid `TrackAnalysis` JSON instead of
decoding and analyzing the entire library again. Failed files are isolated from
successful files. The cache is metadata and analysis; original music is not
modified.

### How a transition is selected

The transition system evaluates a directed pair `current → candidate`, not just a
playlist order. It first scores broad compatibility using confidence-weighted
tempo, harmony, rhythm, energy, structure, vocal, and timbre components. Tempo
supports practical half/double-time relationships; harmony compares tonic/mode
relationships; rhythm compares beat density and inferred meter; energy compares
both global levels and the source exit/candidate entry levels.

For each compatible pair, the transition planner searches multiple source exit
and destination entry windows. It prefers phrase, bar, beat/downbeat, section,
repetition, instrumental, and energy boundaries. Each possible window is checked
for valid duration, available audio, beat alignment, tempo correction, and vocal
activity. The result is a ranked `TransitionPlan` with an exit timestamp, entry
timestamp, duration, strategy, confidence, component reasons, vocal-safety state,
and any bounded beat offset. A candidate with no feasible window is rejected.

The live `PlaybackEngine` uses that plan on two resident decks. It prepares the
incoming audio off the callback path, schedules both decks on a media/sample
clock, and applies a short equal-power fade. The incoming deck begins at its
selected cue rather than assuming 00:00. The outgoing deck leaves at the selected
musical boundary rather than waiting for the file's final sample. The queue and
planner maintain a ten-track lookahead; after five completed songs, another
window is prepared. Seek, reorder, cue, and manual exit changes invalidate future
plans and rebuild them without changing the active audio callback.

The planner is vocal-aware only when it has trusted separated-stem evidence. It
rejects sustained significant vocal-vocal overlap in normal automatic playback,
but uncertain or unavailable vocal data cannot certify safety. The Demucs path
uses RMS activity, absolute/relative floors, attack context, hysteresis, and
release on the separated vocal stem. Its indicators are explicitly uncalibrated
and confidence-limited; they are not a claim of perfect singing detection.

### Set planning, rendering, and optional ML

`set_planning` performs deterministic bounded beam search over eligible future
transitions. It can follow build, maintain, release, or peak energy trajectories,
avoid repeats, discourage recent artist repetition, and apply a small diversity
penalty. Its fixed objective is 65% transition score, 20% trajectory fit, and 15%
variety.

`rendering` is an evaluation/preview path. It executes an already-selected plan,
pitch-preserving-stretches within the permitted correction range, applies bounded
RMS gain matching and peak protection, and writes a float WAV preview. Rendering
does not select tracks and is not used to generate live transition clips.

The optional `transition_quality` model is a
`HistGradientBoostingRegressor` trained on versioned transition features. The
dataset pipeline writes deterministic train/validation/test JSONL splits and
supports group-safe splits and human/consensus annotations. Automatically
generated weak labels come from the deterministic plan score, so they cannot
prove that ML is better than that baseline. The evaluator reports MAE, RMSE,
Pearson correlation, ranking metrics, and diagnostic guardrails. A model is only
appropriate for live use after diverse, leakage-safe, human-labeled holdout
evaluation; until then the live system remains deterministic.

### Runtime data flow

```text
music/ → scanner/loader → DSP analysis → AnalysisCache
                         ↓
             compatibility + transition search
                         ↓
               queue / ten-track lookahead
                         ↓
       two-deck preparation → sample-clock mixer → output device
```

The callback only copies already-prepared PCM and applies scheduled gains. It
does not perform analysis, file I/O, model inference, logging, or blocking work.
Planning runs in a separate process and tempo preparation in a worker, so a slow
analysis job cannot intentionally insert silence into the active stream. Python
audio on a general-purpose operating system remains soft real time: OS/device
stalls can still cause underruns, which are counted and reported.

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
## Live local playback

The core now includes a live two-deck engine, queue, controls and buffered audio
output. Run `python -m ai_dj play music --vocal-stems` after installing
`pip install -e '.[playback,vocals]'`. No transition WAV is generated by this path.
See [live engine architecture, controls and validation](docs/LIVE_AUDIO_ENGINE.md)
for setup, vocal uncertainty and current memory limits.

## Local live DJ workstation

Run `.\.venv\Scripts\python.exe -m ai_dj ui` and open http://127.0.0.1:8765.
The minimal dark UI connects directly to PlaybackEngine, with cached audio waveforms,
cue editing, a live transition timeline, queue drag-and-drop and music imports.
See [workstation controls and validation](docs/WORKSTATION.md).
