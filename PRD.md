# PRD.md — AI DJ: Intelligent Music Mixing & Transition Engine

**Project Name:** AI DJ
**Working Codename:** SYNC
**Version:** 2.0
**Status:** Product + Engineering Source of Truth
**Primary Objective:** Build a generalizable AI-powered DJ engine that can analyze arbitrary songs, understand their musical characteristics, select compatible tracks, identify optimal transition points, and automatically generate musically coherent DJ transitions and complete DJ sets.

---

# 1. Vision

The goal of this project is to build an **autonomous DJ intelligence engine**, not simply a music player, audio editor, or BPM matcher.

The system should accept a collection of previously unseen songs and determine:

1. What each song sounds like and how it is structured.
2. Which song should come next.
3. Where the current song should transition.
4. Where the next song should enter.
5. How their tempos should be aligned.
6. Whether their musical/harmonic characteristics are compatible.
7. Whether vocals should overlap or be avoided.
8. How the energy of the set should evolve.
9. How the transition should actually be rendered.

The ultimate goal is:

> **Given arbitrary music, autonomously produce DJ transitions that sound intentional and musically natural.**

The system must generalize beyond the songs used during development or training.

---

# 2. Core Product Definition

AI DJ consists of three major intelligence/processing layers:

```text
                 USER MUSIC
                     │
                     ▼
          ┌─────────────────────┐
          │ MUSIC UNDERSTANDING │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │     AI DJ BRAIN     │
          │                     │
          │ Song Selection      │
          │ Transition Quality  │
          │ Transition Timing   │
          │ Set Planning        │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │   AUDIO ENGINE      │
          │                     │
          │ Beat Matching       │
          │ Time Stretching     │
          │ EQ                  │
          │ Crossfading         │
          │ Stem Mixing         │
          └──────────┬──────────┘
                     │
                     ▼
                DJ SET OUTPUT
```

---

# 3. Generalization Requirement

This is a **fundamental requirement**.

The system must not be designed around a fixed collection of songs.

The system should work on songs that were not present in the training dataset.

For example:

```text
Training data:
    thousands of songs / transitions

User library:
    completely different songs

Expected:
    model analyzes unseen songs
    ↓
    extracts their representations
    ↓
    predicts compatibility
    ↓
    plans transitions
```

The system must not memorize filenames, artists, or individual songs as a substitute for understanding musical characteristics.

---

# 4. What This Project Is NOT

The project is NOT:

* A simple MP3 player
* A playlist randomizer
* A BPM sorter
* A normal crossfade tool
* A fixed rule-based playlist generator
* A model trained on 3–5 songs
* A system that only works on one genre
* A system that only works on one language
* A UI-first project
* A WebGL/WebXR project at this stage

The core product is the **music intelligence + transition engine**.

---

# 5. Primary Product Goals

The final system should:

### G1 — Understand Music

Extract meaningful musical representations from arbitrary audio.

### G2 — Generalize

Work on previously unseen songs.

### G3 — Select Songs

Determine which song is a good candidate to follow another.

### G4 — Select Transition Points

Determine where each song should enter and exit.

### G5 — Generate High-Quality Transitions

Render musically coherent transitions.

### G6 — Generate Complete Sets

Plan an entire sequence rather than greedily selecting only the next track.

### G7 — Learn Transition Quality

Train/evaluate machine-learning models that learn what constitutes a good transition.

---

# 6. Fundamental Design Principle

The project combines:

```text
Music Information Retrieval
+
Digital Signal Processing
+
Machine Learning
+
Optimization
+
Audio Rendering
```

No single technique is expected to solve the entire problem.

---

# 7. System Architecture

The intended long-term architecture is:

```text
                         AUDIO FILES
                              │
                              ▼
                     ┌────────────────┐
                     │ Audio Ingestion│
                     └───────┬────────┘
                             │
                             ▼
                   ┌────────────────────┐
                   │ Music Understanding│
                   └─────────┬──────────┘
                             │
              ┌──────────────┼───────────────┐
              ▼              ▼               ▼
            Rhythm         Harmony         Structure
              │              │               │
              ▼              ▼               ▼
             BPM            Key            Sections
             Beats          Chroma          Phrases
             Bars           Tonality        Energy
              │              │               │
              └──────────────┼───────────────┘
                             ▼
                    Unified Track
                    Representation
                             │
                             ▼
                 ┌──────────────────────┐
                 │   AI DJ Intelligence  │
                 │                      │
                 │ Song Ranking         │
                 │ Transition Scoring   │
                 │ Entry/Exit Selection │
                 │ Set Planning         │
                 └──────────┬───────────┘
                            │
                            ▼
                  Transition Specification
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Audio Engine      │
                 │                      │
                 │ Time Stretch         │
                 │ Beat Alignment       │
                 │ EQ                   │
                 │ Crossfade            │
                 │ Stems                │
                 └──────────┬───────────┘
                            │
                            ▼
                       FINAL DJ SET
```

---

# 8. Music Understanding Layer

The Music Understanding Layer converts raw audio into structured information.

It should eventually analyze:

## Rhythm

* BPM
* Tempo confidence
* Beat positions
* Downbeats
* Bars
* Time signature where possible
* Rhythmic descriptors

## Harmony

* Key
* Mode
* Chroma
* Harmonic descriptors
* Harmonic compatibility

## Structure

* Intro
* Verse
* Chorus
* Drop
* Breakdown
* Bridge
* Outro
* Repeating sections
* Phrase boundaries

## Energy

* Global energy
* Local energy
* Energy trajectory
* Spectral intensity
* Rhythmic intensity

## Vocals

* Vocal probability
* Vocal/instrumental regions
* Vocal density

## Timbre

* Spectral features
* MFCC
* Spectral centroid
* Spectral contrast
* Other useful descriptors

## Melody

* Pitch contour
* Melodic representation
* Melodic similarity

## Advanced Indian Music Features

Eventually investigate:

* Tonic
* Swara-related representation
* Pitch movement
* Characteristic melodic phrases
* Raga-related features

Raga analysis is an experimental research component and must not be presented as reliable unless properly validated.

---

# 9. Unified Track Representation

Each song must eventually be represented as structured data.

Example:

```json
{
  "track_id": "unique-id",
  "path": "song.mp3",
  "duration": 241.82,

  "tempo": {
    "bpm": 128.0,
    "confidence": 0.94
  },

  "rhythm": {
    "beats": [],
    "downbeats": [],
    "time_signature": "4/4"
  },

  "harmony": {
    "key": "A minor",
    "confidence": 0.84
  },

  "energy": {
    "global": 0.81,
    "timeline": []
  },

  "structure": {
    "sections": [],
    "phrases": []
  },

  "vocals": {
    "present": true,
    "timeline": []
  },

  "timbre": {},
  "melody": {},
  "raga": {},

  "analysis_version": "1.0"
}
```

The schema should evolve without breaking the entire architecture.

---

# 10. Training Data vs User Data

This distinction is mandatory.

## Training Data

Used to train and validate AI models.

Potentially:

```text
thousands to millions of:
    tracks
    track pairs
    transition candidates
    transition quality labels
```

## User Data

Used during inference.

Example:

```text
user/
    songA.mp3
    songB.mp3
    songC.mp3
```

These songs may never have been seen by the model during training.

The model should analyze their features and make predictions based on learned musical relationships.

---

# 11. 3–5 Songs Are NOT Training Data

A small local collection of 3–5 songs is only for:

* Development
* Debugging
* Integration testing
* Manual listening tests

It must NOT be treated as the ML training dataset.

---

# 12. Machine Learning Objective

The central ML research problem is:

> **Learn to estimate the quality of a potential transition between two songs.**

A candidate input may contain:

```text
Track A representation
+
Track B representation
+
Track A transition region
+
Track B entry region
+
relationship features
```

The model produces:

```text
transition_quality_score
```

For example:

```text
0.94 = highly promising
0.72 = acceptable
0.31 = poor
```

The score represents a ranking signal, not an absolute measurement of musical truth.

---

# 13. ML Architecture

The exact model architecture must be determined experimentally.

Potential progression:

```text
Baseline:
Feature vectors
     ↓
Regression / ranking model
```

Then:

```text
Advanced:
Neural network
     ↓
Track embeddings
     +
Transition features
     ↓
Quality prediction
```

Later:

```text
Advanced:
Learned music embeddings
+
pairwise ranking
+
sequence optimization
```

Do not begin with a complex architecture.

Establish a strong baseline first.

---

# 14. ML Input Features

Potential features include:

### Tempo

```text
BPM_A
BPM_B
BPM_difference
BPM_ratio
tempo_adjustment_required
```

### Harmony

```text
key_A
key_B
harmonic_distance
key_compatibility
```

### Rhythm

```text
beat_alignment
rhythm_similarity
downbeat_alignment
```

### Structure

```text
section_A
section_B
phrase_alignment
intro/outro compatibility
```

### Energy

```text
energy_A
energy_B
energy_difference
energy_slope
```

### Vocals

```text
vocal_A
vocal_B
vocal_overlap
instrumental_window_overlap
```

### Spectral

```text
spectral_similarity
timbre_similarity
```

### Melody

```text
melodic_similarity
pitch compatibility
```

---

# 15. Training Target

Possible labels:

```text
transition_quality ∈ [0, 1]
```

or ranking labels:

```text
A > B
```

For example:

```text
Transition 1 = 0.93
Transition 2 = 0.81
Transition 3 = 0.44
Transition 4 = 0.17
```

Ranking-based learning should be investigated because transition quality is inherently comparative.

---

# 16. Dataset Generation

A major part of this project is **building the dataset**.

The system should eventually support automatic generation of transition candidates.

Conceptually:

```text
Track A
   ×
Track B
   ×
Candidate exit points
   ×
Candidate entry points
```

produces:

```text
Transition Candidates
```

Each candidate can be evaluated using:

* DSP metrics
* Musical compatibility
* Structural alignment
* Audio quality
* Human evaluation

---

# 17. Weak Supervision

Initially, it may be impractical to manually label millions of transitions.

Therefore investigate weak supervision.

Potential automatic signals:

```text
beat alignment
phrase alignment
harmonic compatibility
energy continuity
vocal collision
spectral compatibility
audio artifacts
```

These can produce preliminary quality estimates.

However:

> Automatically generated labels must not automatically be treated as ground truth.

Human evaluation must be used to validate the labeling strategy.

---

# 18. Human Evaluation

Human listening tests are essential.

A sample evaluation could be:

```text
Transition:
A → B

Beat alignment:       1–5
Musical compatibility:1–5
Energy flow:          1–5
Vocal interaction:    1–5
Overall quality:      1–5
```

The dataset should eventually contain a high-quality human-validated subset.

---

# 19. Dataset Diversity

The training/evaluation data must be diverse.

The system should eventually be tested across:

* Electronic
* Pop
* Hip-hop
* Rock
* Classical
* Indian music
* K-pop
* J-pop
* Lo-fi
* Ambient
* Instrumental
* Acoustic
* Different languages
* Different production styles
* Different recording qualities
* Different BPM ranges

The goal is generalization, not specialization.

---

# 20. Data Leakage Prevention

This is critical.

Training and test sets must avoid trivial leakage.

For example:

```text
same song
```

must not appear in both training and test sets in a way that allows memorization.

Where appropriate, splits should consider:

* Artist
* Album
* Song
* Dataset source
* Genre

The test set should measure performance on genuinely unseen material.

---

# 21. Generalization Evaluation

The model must be evaluated on music that was not used for training.

Important evaluation categories:

```text
Known genre / unseen song
Unseen artist
Unseen album
Unseen genre
Unseen language
Mixed-genre playlists
Low-confidence analysis
```

This will reveal whether the system actually learned useful musical relationships.

---

# 22. Song Selection

The AI DJ must eventually rank possible next tracks.

For:

```text
Current Track A
```

the system generates:

```text
Candidate B
Candidate C
Candidate D
Candidate E
...
```

and computes compatibility.

Eventually:

```text
Track representation
+
candidate representation
+
context
        ↓
ML compatibility model
        ↓
ranking
```

---

# 23. Transition Candidate Search

For the chosen next track:

```text
Track A
    ↓
possible exit points

Track B
    ↓
possible entry points
```

The engine evaluates combinations.

Example:

```text
A exit:
02:32
02:48
03:04
03:20

B entry:
00:00
00:16
00:32
00:48
```

Each combination becomes a transition candidate.

---

# 24. Transition Planning

The transition planner must determine:

```text
source track
destination track
source exit point
destination entry point
transition duration
target BPM
time-stretch ratio
beat alignment
phrase alignment
mix strategy
```

Example:

```json
{
  "source": "songA",
  "destination": "songB",
  "source_exit": 208.4,
  "destination_entry": 16.2,
  "duration": 16.0,
  "target_bpm": 127.0,
  "strategy": "phrase_crossfade"
}
```

---

# 25. Beat Matching

The engine must align rhythmic grids.

Example:

```text
Song A:
| 1 2 3 4 | 1 2 3 4 |

Song B:
| 1 2 3 4 | 1 2 3 4 |
            ↑
         aligned
```

Tempo changes should use pitch-preserving time stretching where possible.

---

# 26. Phrase Matching

Transitions should preferably occur at musical phrase boundaries.

Preferred:

```text
Song A
| phrase | phrase | phrase |
                    ↓
                 transition

Song B
| phrase | phrase |
↑
entry
```

The engine should avoid arbitrary beat-level cuts when better phrase-level opportunities exist.

---

# 27. Vocal-Aware Mixing

The system should understand vocal regions.

Preferred:

```text
A vocal
    ↓
A instrumental
    ↓
B instrumental
    ↓
B vocal
```

Avoid unnecessary:

```text
A vocal
+
B vocal
```

overlap.

Intentional vocal overlap may be allowed in advanced stem-mixing modes.

---

# 28. Energy-Aware DJing

The engine should model energy as a trajectory.

Possible set trajectories:

```text
Build:
0.4 → 0.5 → 0.7 → 0.9

Maintain:
0.8 → 0.82 → 0.79 → 0.81

Release:
0.9 → 0.75 → 0.55
```

The system should avoid uncontrolled energy jumps unless explicitly requested.

---

# 29. Audio Rendering Engine

The rendering layer must be independent from the ML layer.

Its responsibilities include:

* Time stretching
* Beat alignment
* Gain adjustment
* Crossfade
* EQ
* Filtering
* Loudness management
* Optional stem mixing

The ML system decides **what should happen**.

The audio engine decides **how to physically render it**.

---

# 30. Stem Mixing

Advanced versions may separate:

```text
Vocals
Drums
Bass
Other
```

This enables:

```text
A drums
+
A bass
+
B melody
+
B vocals
```

Stem separation should not block the initial product.

---

# 31. Melody Intelligence

Eventually extract:

```text
pitch contour
+
melodic embeddings
+
melodic similarity
```

This may improve compatibility beyond traditional key matching.

---

# 32. Raga Intelligence

Raga analysis is a long-term research feature.

Potential pipeline:

```text
Audio
 ↓
Tonic detection
 ↓
Pitch extraction
 ↓
Pitch normalization
 ↓
Melodic representation
 ↓
Raga-related analysis
```

This must be evaluated separately from Western key detection.

The system must represent uncertainty.

Example:

```text
Estimated raga:
X

Confidence:
0.63
```

rather than presenting uncertain classifications as facts.

---

# 33. Global DJ Set Planning

The final system should not simply select:

```text
best next song
```

It should eventually optimize:

```text
Song A
 ↓
Song B
 ↓
Song C
 ↓
Song D
 ↓
Song E
```

based on the entire sequence.

Possible optimization techniques:

* Graph search
* Beam search
* Dynamic programming
* Ranking models
* Sequence models
* Optimization algorithms

Advanced ML/optimization should be introduced only after a reliable baseline exists.

---

# 34. Avoiding Repetition

A generated set should track previously used tracks.

Unless explicitly configured otherwise:

```text
same song should not repeat
```

within a single set.

---

# 35. Explainability

The engine should explain decisions.

Example:

```text
Selected Song B

BPM compatibility:       0.94
Key compatibility:       0.91
Rhythm similarity:       0.89
Energy continuity:       0.95
Vocal compatibility:     0.87
Transition prediction:   0.93

Reason:
Strong rhythmic and harmonic compatibility with
a suitable instrumental entry window.
```

ML predictions should not become an opaque black box wherever useful explanations can be provided.

---

# 36. Confidence

Every uncertain analysis or prediction should expose confidence where practical.

Examples:

```text
BPM confidence
Key confidence
Vocal confidence
Section confidence
Transition confidence
Raga confidence
```

The system should be able to respond:

```text
confidence too low
```

rather than inventing certainty.

---

# 37. Model Versioning

ML models must be versioned.

Example:

```text
transition_model_v1
transition_model_v2
```

Track analysis should also include:

```text
analysis_version
```

This allows reproducibility and proper evaluation.

---

# 38. Experiment Tracking

ML experiments should record:

```text
model version
dataset version
feature configuration
hyperparameters
training metrics
validation metrics
test metrics
```

Do not overwrite experiment results.

---

# 39. Evaluation Metrics

Possible ML metrics:

### Regression

* MAE
* RMSE
* Correlation

### Ranking

* Spearman correlation
* Kendall correlation
* Pairwise accuracy
* NDCG

### System-level

* Transition acceptance rate
* Human preference rate
* Beat alignment accuracy
* Vocal collision rate
* Loudness consistency
* Artifact rate

Human listening quality remains the most important final evaluation.

---

# 40. Baseline Before ML

Before training an ML model, implement a deterministic baseline.

Example:

```text
BPM compatibility
+
key compatibility
+
rhythm compatibility
+
energy compatibility
+
vocal compatibility
+
phrase alignment
```

This creates a baseline against which the ML model can be compared.

The ML model must demonstrate measurable improvement.

---

# 41. No Fake AI

Do not call a deterministic formula an "AI model."

Do not call a pretrained feature extractor the project's own trained DJ model.

Clearly distinguish:

```text
DSP analysis
Pretrained model
Our trained model
Rule-based scoring
Optimization
```

---

# 42. No ML for Everything

Not every component needs ML.

Use deterministic DSP where it is more reliable.

Examples:

```text
Audio decoding
Time stretching
Crossfading
Beat alignment
Gain calculations
```

ML is most valuable for tasks where musical judgment is difficult to encode explicitly.

---

# 43. Generalization Over Memorization

The model must learn relationships such as:

```text
similar rhythm
compatible harmony
compatible energy
compatible structure
```

rather than:

```text
Song A → Song B
```

The latter is memorization and is not the project's objective.

---

# 44. CLI-First Architecture

The core engine must work without a graphical interface.

Example:

```bash
python -m ai_dj analyze ./music
```

Then:

```bash
python -m ai_dj recommend ./music/song.mp3
```

Eventually:

```bash
python -m ai_dj generate \
    --input ./music \
    --output ./output/set.wav
```

The exact commands may evolve.

---

# 45. API Boundary

The architecture should eventually expose programmatic APIs such as:

```text
analyze_track()
analyze_library()
find_compatible_tracks()
generate_transition()
generate_set()
```

These APIs must not depend on a UI.

---

# 46. Project Structure

Recommended architecture:

```text
ai-dj/
│
├── PRD.md
├── README.md
├── pyproject.toml
├── .gitignore
│
├── src/
│   └── ai_dj/
│       │
│       ├── ingestion/
│       │
│       ├── analysis/
│       │   ├── tempo.py
│       │   ├── beats.py
│       │   ├── harmony.py
│       │   ├── energy.py
│       │   ├── structure.py
│       │   ├── vocals.py
│       │   ├── timbre.py
│       │   ├── melody.py
│       │   └── raga.py
│       │
│       ├── representation/
│       │
│       ├── datasets/
│       │
│       ├── models/
│       │
│       ├── matching/
│       │
│       ├── transition/
│       │
│       ├── rendering/
│       │
│       ├── evaluation/
│       │
│       ├── pipeline/
│       │
│       └── cli/
│
├── tests/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── annotations/
│   └── cache/
│
├── experiments/
│
├── models/
│
└── output/
```

This structure may evolve, but separation of concerns is mandatory.

---

# 47. Recommended Technology

Initial language:

```text
Python
```

Potential libraries:

```text
NumPy
SciPy
librosa
soundfile
FFmpeg
PyTorch
```

Additional dependencies may be introduced when justified.

Do not blindly install every possible music/AI library.

---

# 48. Audio Format Support

Initial support:

```text
MP3
WAV
FLAC
```

The architecture should allow additional formats later.

---

# 49. Caching

Audio analysis is expensive.

The system must cache:

```text
track analysis
features
embeddings
```

Cache invalidation should account for:

* File identity
* File modification
* Analysis version
* Feature version
* Model version where applicable

---

# 50. Reproducibility

Experiments must be reproducible.

Record:

```text
dataset version
model version
configuration
random seed
software versions
```

Where randomness is used, it must be explicitly controlled.

---

# 51. Performance

The system should eventually handle:

```text
100+
1000+
```

tracks efficiently.

Avoid repeatedly processing unchanged files.

For large libraries, precomputed representations and efficient candidate retrieval should be investigated.

---

# 52. Audio Quality

Generated output must avoid:

* Clipping
* Sudden loudness jumps
* Severe beat misalignment
* Obvious tempo artifacts
* Unnecessary vocal collisions
* Severe time-stretch artifacts
* Abrupt spectral changes

Audio quality is a first-class requirement.

---

# 53. Legal / Dataset Requirements

Music used for training and testing must respect applicable licensing and copyright requirements.

Do not commit copyrighted music into the repository.

Prefer:

* Licensed datasets
* Public-domain audio
* Creative Commons audio
* Self-created audio
* Synthetic test signals

Dataset provenance must be documented.

---

# 54. Security / Privacy

User music is local data.

The initial architecture should not require uploading the user's music to a remote service.

Local processing is preferred.

If external APIs/models are introduced later, their data requirements must be explicitly documented.

---

# 55. Development Phases

## Phase 0 — Foundation

Build:

* Python package
* Configuration
* Logging
* CLI
* Testing infrastructure
* Audio ingestion

---

## Phase 1 — Music Analysis

Build:

* BPM
* Beats
* Downbeats
* Key
* Energy
* Spectral features

Deliverable:

```text
audio
→ structured music representation
```

---

## Phase 2 — Musical Structure

Build:

* Bars
* Phrases
* Sections
* Vocal regions

Deliverable:

```text
track
→ musically structured representation
```

---

## Phase 3 — Deterministic DJ Baseline

Build:

* Song compatibility
* Candidate transition search
* Rule-based scoring
* Explainable ranking

Deliverable:

```text
Song A
→ best candidate Song B
→ best transition position
```

This baseline is mandatory before advanced ML.

---

## Phase 4 — Transition Rendering

Build:

* Time stretching
* Beat alignment
* Phrase alignment
* Crossfade
* Gain control

Deliverable:

```text
Song A + Song B
→ rendered transition
```

---

## Phase 5 — Dataset Pipeline

Build:

* Dataset ingestion
* Candidate transition generation
* Feature extraction
* Automatic labeling
* Human annotation format
* Dataset versioning
* Train/validation/test splits

Deliverable:

```text
large transition dataset
```

---

## Phase 6 — First ML Model

Train a baseline transition-quality model.

Input:

```text
Track A features
+
Track B features
+
transition features
```

Output:

```text
transition quality
```

Compare against the deterministic baseline.

---

## Phase 7 — Improved ML DJ Brain

Investigate:

* Learned embeddings
* Pairwise ranking
* Neural transition scoring
* Better candidate retrieval
* Context-aware ranking

---

## Phase 8 — Vocal-Aware DJ

Improve:

* Vocal detection
* Vocal overlap penalties
* Instrumental windows
* Vocal-aware transition planning

---

## Phase 9 — Advanced Mixing

Add:

* EQ transitions
* Frequency-aware mixing
* Better loudness matching
* Improved transition curves

---

## Phase 10 — Stem Mixing

Investigate:

* Vocals
* Drums
* Bass
* Other

Use stems for intelligent mashup-style transitions.

---

## Phase 11 — Melody / Raga

Research:

* Tonic
* Pitch representation
* Melody embeddings
* Raga-related features

This remains experimental.

---

## Phase 12 — Global Set Intelligence

Move from:

```text
best next song
```

to:

```text
best complete sequence
```

Optimize:

* Transition quality
* Energy trajectory
* Musical coherence
* Variety
* Repetition
* Set duration

---

# 56. Milestone 0 Definition of Done

Repository:

```text
✓ installs
✓ package imports
✓ CLI works
✓ tests execute
✓ logging works
✓ configuration works
```

---

# 57. Milestone 1 Definition of Done

Given an arbitrary supported audio file:

```text
✓ file loads
✓ duration detected
✓ BPM estimated
✓ BPM confidence available
✓ beat timestamps generated
✓ analysis serialized
✓ analysis can be reloaded
```

---

# 58. Milestone 2 Definition of Done

Given an arbitrary supported audio file:

```text
✓ BPM
✓ beats
✓ downbeats
✓ key
✓ energy
✓ spectral features
✓ structured representation
```

---

# 59. Milestone 3 Definition of Done

Given:

```text
Song A
+
library of candidate songs
```

the system can:

```text
✓ rank candidates
✓ calculate compatibility
✓ explain scores
✓ identify strongest candidates
```

---

# 60. Milestone 4 Definition of Done

Given:

```text
Song A
+
Song B
```

the system can:

```text
✓ identify transition candidates
✓ align beats
✓ match phrases
✓ choose transition point
✓ render transition
```

---

# 61. ML Definition of Done

The first ML model is not considered successful simply because training completes.

It must:

```text
✓ outperform or meaningfully complement baseline
✓ generalize to unseen songs
✓ have documented evaluation
✓ have reproducible training
✓ have documented limitations
✓ be tested against human judgments
```

---

# 62. Final System Definition of Done

The mature system should accept:

```text
arbitrary local music library
```

and produce:

```text
automatically selected songs
+
automatically selected transition points
+
beat-matched transitions
+
musically coherent progression
+
rendered DJ set
```

with minimal user intervention.

---

# 63. Example Final Workflow

User:

```text
music/
├── song01.mp3
├── song02.mp3
├── song03.mp3
├── song04.mp3
├── song05.mp3
└── ...
```

Runs:

```bash
python -m ai_dj generate --input ./music
```

System:

```text
Scanning library...
Found 87 tracks.

Analyzing tracks...
████████████████████ 100%

Building music representations...

Finding compatible tracks...

Planning DJ sequence...

1. Song 01
2. Song 42
3. Song 17
4. Song 63
5. Song 09

Planning transitions...

Song 01 → Song 42
Quality: 0.94

Song 42 → Song 17
Quality: 0.91

Song 17 → Song 63
Quality: 0.89

Rendering...

output/set.wav
```

---

# 64. Example Transition Reasoning

Suppose:

```text
Song A
BPM: 128
Key: A minor
Energy: 0.82

Song B
BPM: 127
Key: A minor
Energy: 0.79
```

The system identifies:

```text
Song A:
03:28
instrumental phrase ending

Song B:
00:16
instrumental phrase beginning
```

Candidate score:

```text
Tempo:             0.97
Harmony:           1.00
Rhythm:            0.94
Phrase:            0.92
Energy:            0.95
Vocals:            0.91

Overall:           0.95
```

The audio engine then renders the transition.

---

# 65. Important Architectural Separation

The following boundaries must remain clear:

```text
                 ┌─────────────────┐
                 │ Music Analysis  │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ ML / Intelligence│
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Transition Plan │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Audio Rendering │
                 └─────────────────┘
```

Do not combine these into one monolithic module.

---

# 66. Codex Engineering Rules

Codex must:

1. Read `PRD.md` before implementation.
2. Inspect the existing repository before modifying it.
3. Follow the milestone order.
4. Implement the smallest complete milestone.
5. Add tests with every meaningful feature.
6. Run tests before declaring success.
7. Avoid unnecessary dependencies.
8. Avoid unnecessary architecture changes.
9. Preserve working code.
10. Document important architectural decisions.
11. Never claim an ML model is generalizable without evaluation.
12. Never claim a transition is "perfect" without appropriate evaluation.
13. Never treat 3–5 user songs as an ML training dataset.
14. Never hard-code a fixed song library into the intelligence system.
15. Never implement UI unless explicitly requested.
16. Never introduce ML where deterministic DSP is more appropriate without justification.
17. Keep experimental research components isolated from production-critical components.

---

# 67. Codex Workflow

For every task:

```text
1. Read PRD.md
2. Inspect repository
3. Identify current milestone
4. Inspect relevant existing code
5. Create implementation plan
6. Implement
7. Add tests
8. Run tests
9. Perform manual/integration verification where possible
10. Update documentation
11. Report limitations
```

Do not skip repository inspection.

---

# 68. Current Development Target

The immediate implementation target is:

## Phase 0 + Phase 1

Build:

```text
Repository foundation
        ↓
Audio ingestion
        ↓
BPM detection
        ↓
Beat detection
        ↓
Track representation
        ↓
Analysis cache
        ↓
JSON output
        ↓
Tests
```

Do NOT yet build:

```text
ML training
Song recommendation
Transition rendering
Stem separation
Raga detection
UI
WebGL
WebXR
```

The purpose of these early phases is to establish reliable music representations that the future ML system can learn from.

---

# 69. Future ML Dataset Principle

Once the analysis pipeline is reliable, the next major research task is:

> **Build a large, diverse, legally usable transition dataset.**

The dataset should contain examples representing:

```text
good transitions
acceptable transitions
bad transitions
```

across diverse music.

The dataset must be separated from the user's local music library.

---

# 70. Core Success Metric

The ultimate success metric is NOT:

```text
accuracy on a test CSV
```

It is:

> **When a human listens to a generated transition between previously unseen songs, does it sound like something a competent DJ intentionally created?**

The ML metrics support this objective.

They do not replace it.

---

# 71. Final Product Philosophy

The project should evolve through:

```text
Reliable DSP
      ↓
Music Understanding
      ↓
Strong Deterministic Baseline
      ↓
Large Diverse Dataset
      ↓
ML Transition Intelligence
      ↓
Advanced Audio Rendering
      ↓
Global DJ Set Optimization
```

The goal is not to make the project "look AI."

The goal is to make the system **actually make better musical decisions**.

---

# 72. Absolute Source-of-Truth Rules

When making implementation decisions, use this hierarchy:

```text
1. PRD.md
2. Tests
3. Existing architecture
4. Documented experiment results
5. Implementation details
6. Developer convenience
```

If new research demonstrates that an existing approach is inferior, update the PRD and document the decision rather than silently changing direction.

---

# 73. End State

The completed system should behave conceptually like this:

```text
                 ANY MUSIC LIBRARY
                        │
                        ▼
                AUDIO UNDERSTANDING
                        │
                        ▼
                 MUSIC FEATURES
                        │
                        ▼
                 LEARNED DJ BRAIN
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      NEXT SONG     EXIT POINT     ENTRY POINT
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                 TRANSITION PLAN
                        │
                        ▼
                  AUDIO ENGINE
                        │
                        ▼
                    DJ SET
```

The defining characteristic of the project is:

> **It should make musically informed decisions on music it has never encountered before.**

That is the central requirement around which the entire architecture, dataset strategy, ML strategy, evaluation methodology, and engineering roadmap must be designed.

---

# END OF PRD
