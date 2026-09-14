"""Worker-only stereo decoding and pitch-preserving deck preparation."""
from dataclasses import dataclass
import numpy as np
import soundfile as sf
from collections import OrderedDict
from collections.abc import Mapping
from threading import Lock


class AudioLibrary(Mapping):
    """Bounded native-rate PCM cache. Missing items may be loaded by workers only."""
    def __init__(self, tracks, sample_rate, memory_limit_mb):
        self.tracks = {t.track_id: t for t in tracks}
        self.sample_rate = sample_rate
        self.limit = memory_limit_mb * 1024 ** 2
        self.cache = OrderedDict()
        self.pinned = set()
        self.lock = Lock()
        self.current = None

    def __iter__(self):
        return iter(self.tracks)

    def __len__(self):
        return len(self.tracks)

    def resident(self, track_id):
        if self.current is not None and self.current[0] == track_id:
            return self.current[1]
        return self.cache.get(track_id)

    def __getitem__(self, track_id):
        existing = self.resident(track_id)
        if existing is not None:
            return existing
        if self.tracks[track_id].duration * self.sample_rate * 8 * 4 > self.limit:
            raise ValueError("Track exceeds PCM/tempo preparation budget")
        # This method's miss path belongs exclusively to preparation threads.
        audio = decode_track(self.tracks[track_id].source_path, self.sample_rate)
        if audio.samples.nbytes * 4 > self.limit:
            raise ValueError("Track exceeds PCM/tempo preparation budget")
        with self.lock:
            self.cache[track_id] = audio
            while len(self.cache) > 3 or sum(a.samples.nbytes for a in self.cache.values()) > self.limit / 2:
                victim = next((key for key in self.cache if key not in self.pinned and key != track_id), None)
                if victim is None:
                    break
                del self.cache[victim]
        return audio


@dataclass(frozen=True)
class PreparedAudio:
    samples: np.ndarray  # frames x stereo channels; already gain matched
    rate: float = 1.0
    original_samples: np.ndarray | None = None


def decode_track(path, sample_rate):
    import librosa
    audio, original = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] != 2:
        raise ValueError("Playback supports mono or stereo source audio")
    if original != sample_rate:
        audio = librosa.resample(audio.T, orig_sr=original, target_sr=sample_rate).T
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {path}")
    # One fixed per-track gain persists after the transition; no block-wise AGC jumps.
    rms = max(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))), 1e-8)
    gain = min(10 ** (6 / 20), .16 / rms, .68 / max(float(np.max(np.abs(audio))), 1e-8))
    return PreparedAudio(np.ascontiguousarray(audio * gain, dtype=np.float32))


def prepare_rate(audio, rate):
    if abs(rate - 1.0) < 1e-6:
        return audio
    import librosa
    # Stretch individual tracks, never an assembled transition; mix only at runtime.
    samples = librosa.effects.time_stretch(audio.samples.T, rate=rate).T
    peak = float(np.max(np.abs(samples)))
    if peak > .68:
        samples *= .68 / peak
    return PreparedAudio(np.ascontiguousarray(samples, dtype=np.float32), rate, audio.samples)
