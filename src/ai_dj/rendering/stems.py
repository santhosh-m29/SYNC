"""Optional local Demucs stem separation with file-backed caching."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from ai_dj.ingestion.loader import load_audio
from ai_dj.rendering.models import StemPaths


class StemSeparationError(RuntimeError):
    pass


def stem_cache_root(source_path, cache_directory):
    source = Path(source_path)
    stat = source.stat()
    key = hashlib.sha256(f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|htdemucs-stereo-normalized-v2".encode()).hexdigest()[:24]
    return Path(cache_directory) / key


def separate_stems(source_path: str | Path, cache_directory: str | Path) -> StemPaths:
    """Create/reuse Demucs vocal and accompaniment stems for one track."""
    source = Path(source_path)
    root = stem_cache_root(source, cache_directory)
    vocals, accompaniment = root / "vocals.wav", root / "accompaniment.wav"
    ready = root / "complete-v2"
    if vocals.is_file() and accompaniment.is_file() and ready.is_file():
        return StemPaths(vocals, accompaniment)
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as error:  # pragma: no cover - installation is environment-specific.
        raise StemSeparationError("Install ai-dj[vocals] to enable local vocal stem separation") from error
    model = get_model("htdemucs")
    audio, original_rate = sf.read(source, dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    if audio.shape[1] != 2 or not len(audio) or not np.isfinite(audio).all():
        raise StemSeparationError("Expected finite mono/stereo source audio")
    if original_rate != model.samplerate:
        import librosa
        audio = librosa.resample(audio.T, orig_sr=original_rate, target_sr=model.samplerate).T
    mix = torch.from_numpy(np.ascontiguousarray(audio.T))
    reference = mix.mean(0)
    mean, scale = reference.mean(), reference.std() + 1e-8
    with torch.no_grad():
        separated = (apply_model(model, ((mix - mean) / scale).unsqueeze(0),
                                 device="cpu", progress=False, shifts=0)[0] * scale + mean).cpu().numpy()
    sources = {name: separated[index].T for index, name in enumerate(model.sources)}
    vocal = sources["vocals"].astype(np.float32, copy=False)
    backing = sum((samples for name, samples in sources.items() if name != "vocals"), start=np.zeros_like(vocal)).astype(np.float32)
    root.mkdir(parents=True, exist_ok=True)
    sf.write(vocals, vocal, model.samplerate, subtype="FLOAT")
    sf.write(accompaniment, backing, model.samplerate, subtype="FLOAT")
    ready.write_text("htdemucs stereo normalized v2\n", encoding="utf-8")
    return StemPaths(vocals, accompaniment)
