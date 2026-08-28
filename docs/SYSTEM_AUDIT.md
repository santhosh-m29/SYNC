# System Audit — 2026-08-28

## Scope and baseline

The repository contains modular ingestion, DSP analysis, typed track
representations, structural analysis, deterministic compatibility and transition
planning, datasets, an experimental tabular model, set planning, rendering, and
an end-to-end CLI. The full suite passed before changes: **52 tests**.

A cached six-track local-library generation selected four tracks and three
transitions. Baseline metrics were: 88.0 s output, peak 0.98, RMS 0.171543,
zero clipped samples, 36.844 s planning, 14.964 s rendering, and 51.873 s total.

## Evidence-based findings

- **ML is not promotable.** Dataset v1 has 96 automatic-label examples but zero
  validation and test rows. Its label is the deterministic plan score, so it
  cannot demonstrate superiority over that baseline.
- **Planner work repeated.** Beam states recalculated the same directed track
  pair's transition candidates. This affected latency, not selected plans.
- **Preview assembly had hard joins.** Raw segment boundary jumps were 0.133612
  and 0.063567 (mean 0.098590). These are diagnostics only, not a perceptual
  artifact score.
- **Analysis limitations remain visible.** Vocal activity is unavailable, key
  confidence is modest on the local sample, and tempo/downbeat confidence is
  sometimes low. Existing confidence-aware weights and hard tempo limits are
  retained rather than masked.
- **No valid human preference result exists.** Rendered local previews are not
  a blind listening evaluation.

## Retained baseline

The deterministic compatibility, candidate planner, set objective, and renderer
remain the baseline. No neural model, source separation, or genre-specific rule
was introduced.

## Improvements retained

- Per-generation directed-pair transition memoization in beam search.
- A 20 ms equal-power join between rendered preview segments.
- Final-set quality gate for finite samples and clipping, plus recorded join,
  beat-offset, RMS-range, and unavailable-artifact diagnostics.

## Measured comparison

On the same cached local library with identical configuration, the selected
tracks and transition scores were unchanged. Planning fell to **9.344 s** and
total generation to **16.550 s**. The final set has 87.96 s duration, peak
0.98, RMS 0.171546, zero clipped samples, finite samples, 0.0 s maximum
recorded renderer beat offset, 1.433 dB transition RMS range, and a 20 ms join.

The join metric measures only adjacent samples at the two overlap edges; it does
not treat normal waveform motion inside the crossfade as an artifact. This is an
engineering/technical improvement, not evidence of improved human
musical preference. Human listening and a genuinely held-out labeled dataset
remain required before claiming musical or ML improvement.

## Expanded unseen-library integration run

The final end-to-end run used the now-expanded local library of **32 tracks**.
Six tracks reused existing cache entries and 26 were newly analyzed; no track
failed analysis. With a target of four tracks, a six-item retrieval pool, beam
width six, seed seven, and a maintain trajectory, the system selected:

1. `Jalsa.mp3`
2. `Chola Chola.mp3`
3. `Hukum - Thalaivar Alappara.mp3`
4. `Aasaiye Kaathule.mp3`

It rendered three phrase-crossfade transitions with scores 0.920, 0.914, and
0.921 (mean 0.918333). The final 22,050 Hz float WAV is 95.96 s, has peak 0.98,
RMS 0.132792, zero clipped samples, and finite samples. The renderer reported
at most 0.050 s beat alignment shift, a 6.9714 dB range across transition RMS
values, and 20 ms join edges with maximum adjacent-sample jump 0.038611.

Timing was 149.043 s analysis, 11.049 s retrieval/planning, 4.886 s rendering,
and 165.023 s total. This is a real pipeline run on local input, but it is not
a controlled benchmark and has no human listening result.
