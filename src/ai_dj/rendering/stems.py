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


def separate_stems(source_path: str | Path, cache_directory: str | Path) -> StemPaths:
    """Create/reuse Demucs vocal and accompaniment stems for one track."""
    source = Path(source_path)
    stat = source.stat()
    key = hashlib.sha256(f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|htdemucs".encode()).hexdigest()[:24]
    root = Path(cache_directory) / key
    vocals, accompaniment = root / "vocals.wav", root / "accompaniment.wav"
    if vocals.is_file() and accompaniment.is_file():
        return StemPaths(vocals, accompaniment)
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as error:  # pragma: no cover - installation is environment-specific.
        raise StemSeparationError("Install ai-dj[vocals] to enable local vocal stem separation") from error
    model = get_model("htdemucs")
    audio = load_audio(source, sample_rate=model.samplerate).samples
    mix = torch.from_numpy(np.stack((audio, audio))).unsqueeze(0)
    with torch.no_grad():
        separated = apply_model(model, mix, device="cpu", progress=False)[0].cpu().numpy()
    sources = {name: separated[index].mean(axis=0) for index, name in enumerate(model.sources)}
    vocal = sources["vocals"].astype(np.float32, copy=False)
    backing = sum((samples for name, samples in sources.items() if name != "vocals"), start=np.zeros_like(vocal)).astype(np.float32)
    root.mkdir(parents=True, exist_ok=True)
    sf.write(vocals, vocal, model.samplerate, subtype="FLOAT")
    sf.write(accompaniment, backing, model.samplerate, subtype="FLOAT")
    return StemPaths(vocals, accompaniment)
