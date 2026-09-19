"""Persistent private library, worker analysis, and existing transition planning."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from threading import Lock
import json
import sqlite3
import uuid
import hashlib
import secrets


def analyze(source, root, vocals):
    from ai_dj.ui.server import analyze_import, waveform
    from ai_dj.playback.audio import decode_track
    import soundfile as sf
    track = analyze_import(source, root)
    if vocals:
        from ai_dj.analysis.vocals import analyze_track_vocals
        track = replace(track, structure=replace(track.structure,
            vocal_activity=analyze_track_vocals(source, Path(root) / "stems")))
    peaks = waveform(source, Path(root) / "waveforms")
    output = Path(source).with_suffix(".playback.wav")
    audio = decode_track(source, 44100)
    sf.write(output, audio.samples, 44100, subtype="PCM_16")
    return track.to_dict(), peaks, str(output)


def plan(analyses, current, candidates, position, rate, cue, point):
    from ai_dj.representation.track import TrackAnalysis
    from ai_dj.playback.planner import choose_next, plan_event
    tracks = {key: TrackAnalysis.from_dict(value) for key, value in analyses.items()}
    source = tracks[current]
    target = tracks[candidates[0]] if len(candidates) == 1 else choose_next(source, [tracks[t] for t in candidates])
    event = plan_event(source, target, position, rate, cue=cue.get(target.track_id), transition=point)
    if point is not None or target.track_id in cue:
        event = replace(event, strategy="manual_" + event.strategy)
    return asdict(event)


def stretch(path, root, track_id, rate):
    import soundfile as sf
    from ai_dj.playback.audio import PreparedAudio, prepare_rate
    import numpy as np
    output = Path(root) / f"{track_id}-{rate:.6f}.wav"
    if not output.exists():
        samples, sr = sf.read(path, dtype="float32", always_2d=True)
        prepared = prepare_rate(PreparedAudio(np.ascontiguousarray(samples)), rate)
        sf.write(output, prepared.samples, sr, subtype="PCM_16")
    return str(output)


class Library:
    def __init__(self, root, vocals=False):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.vocals = vocals
        self.pool = ProcessPoolExecutor(max_workers=1)
        self.futures = {}
        self.lock = Lock()
        with self.db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS tracks (id TEXT PRIMARY KEY, filename TEXT, source TEXT, status TEXT, error TEXT, analysis TEXT, peaks TEXT, audio TEXT)")
            if 'owner' not in {row['name'] for row in db.execute('PRAGMA table_info(tracks)')}:
                db.execute('ALTER TABLE tracks ADD COLUMN owner TEXT')
            db.execute('CREATE TABLE IF NOT EXISTS sessions (owner TEXT PRIMARY KEY)')
            interrupted = db.execute("SELECT id, source FROM tracks WHERE status='analyzing'").fetchall()
        for row in interrupted:
            self.submit(row["id"], row["source"])

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.root / "library.sqlite", timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def submit(self, key, source):
        future = self.pool.submit(analyze, source, str(self.root), self.vocals)
        def done(job):
            try:
                analysis, peaks, audio = job.result()
                # Public IDs remain stable independently of the container path.
                analysis["track_id"] = key
                with self.db() as db:
                    db.execute("UPDATE tracks SET status='ready', analysis=?, peaks=?, audio=? WHERE id=?",
                               (json.dumps(analysis), json.dumps(peaks), audio, key))
            except Exception as error:
                with self.db() as db:
                    db.execute("UPDATE tracks SET status='error', error=? WHERE id=?", (str(error), key))
        future.add_done_callback(done)

    def register(self, key, filename, source, owner=None):
        with self.db() as db:
            db.execute("INSERT INTO tracks(id, filename, source, status, owner) VALUES(?,?,?,'analyzing',?)",
                       (key, filename, str(source), owner))
        self.submit(key, str(source))

    def row(self, key):
        with self.db() as db:
            row = db.execute("SELECT * FROM tracks WHERE id=?", (key,)).fetchone()
        if row is None:
            raise KeyError("Unknown track")
        return dict(row)

    def list(self, owner=None):
        with self.db() as db:
            rows = db.execute("SELECT * FROM tracks WHERE owner IS ? ORDER BY rowid", (owner,)).fetchall()
        result = []
        for row in rows:
            info = json.loads(row["analysis"]) if row["analysis"] else None
            result.append(dict(id=row["id"], filename=row["filename"], title=Path(row["filename"]).stem,
                status=row["status"], error=row["error"], duration=info["duration"] if info else 0,
                bpm=info["tempo"]["bpm"] if info else 0, key=info["key"]["key"] if info else None,
                vocals=info["structure"]["vocal_activity"] if info else None))
        return result

    def prepare(self, key, rate):
        row = self.row(key)
        if row["status"] != "ready":
            raise ValueError("Track analysis is not ready")
        if rate == 1:
            return row["audio"]
        cache_key = (key, rate)
        with self.lock:
            if cache_key not in self.futures:
                self.futures[cache_key] = self.pool.submit(stretch, row["audio"], str(self.root), key, rate)
            future = self.futures[cache_key]
        return future.result() if future.done() else None

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def session(self):
        token = secrets.token_urlsafe(32)
        owner = hashlib.sha256(token.encode()).hexdigest()
        with self.db() as db:
            db.execute('INSERT INTO sessions(owner) VALUES(?)', (owner,))
        return token

    def scoped(self, token):
        owner = hashlib.sha256(token.encode()).hexdigest()
        with self.db() as db:
            exists = db.execute('SELECT 1 FROM sessions WHERE owner=?', (owner,)).fetchone()
        if not exists:
            raise KeyError('Unknown session')
        return SessionLibrary(self, owner)


class SessionLibrary:
    """Every HTTP operation is scoped to one anonymous browser library."""
    def __init__(self, library, owner):
        self.library, self.owner = library, owner
        self.pool = library.pool

    def list(self):
        return self.library.list(self.owner)

    def row(self, key):
        row = self.library.row(key)
        if row['owner'] != self.owner:
            raise KeyError('Unknown track')
        return row

    def register(self, key, filename, source):
        self.library.register(key, filename, source, self.owner)

    def prepare(self, key, rate):
        self.row(key)
        return self.library.prepare(key, rate)
