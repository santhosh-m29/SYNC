# Vocal-Aware Transition Planning

## Hard playback gate

Normal playback is fail-closed. A transition is eligible only when both tracks
have trusted vocal timelines and the time-stretched, beat-shifted overlap has
zero sustained significant vocal-vocal collision. ML scores do not override
this gate. Missing or low-confidence vocal evidence therefore produces **no
approved crossfade**, rather than an unverified claim of vocal safety. In that
case, the planner may use a phrase/downbeat-aligned `hard_handoff` with exactly
zero overlap, which is safe by construction and preserves the distinction
between track-level vocal presence and simultaneous audible vocal content.

Rejected plans are retained exclusively for offline dataset hard-negative
generation; they are not returned by the normal transition-planning API.

## Capability boundary

The current local analyzer deliberately returns `vocal_activity.available =
false` for ordinary mixed audio. Generic spectral DSP cannot reliably separate
singing from melodic instruments, so it would be misleading to invent vocal
probabilities for the local MP3 library.

The planner therefore uses vocal-aware behavior only when a trusted
`VocalActivityEstimate` is supplied by a validated external activity detector
or controlled fixture. A timeline is represented as piecewise probability
segments with `start`, `end`, and `probability`.

## Planning behavior

For an aligned source/destination overlap, the planner computes the mean
integral of `source_probability × destination_probability`. It marks plans as
`safe`, `reject`, or `unverifiable`. Any sustained significant collision is
rejected. If every candidate collides, no normal overlapping transition is
returned rather than inventing an arbitrary cut.

Every plan records `incoming_vocal_start` when known and a `vocal_safety`
classification. An instrumental entry is selected when the incoming track has a
meaningful later vocal entrance. Beat, phrase, tempo, harmony, energy, and
technical constraints remain in force.

## Renderer safety

The renderer repeats the hard timeline check after applying its actual tempo
stretch and beat offset, before it writes a WAV. It refuses the render if that
mapped overlap is unsafe or unverifiable. It does not derive stems or claim
lead-vocal knowledge from a mixed file; a future stem renderer can provide a
stronger source-level validation path.

## Evaluation status

Automated trusted-timeline tests demonstrate zero collision scores for
non-overlapping regions, strong penalties for coincident vocal regions, safe
candidate preference, and preserved beat/phrase plans. No real-library
before/after vocal metric or listening preference is claimed because the current
local audio analyses do not have validated vocal activity data.
