"""Loopback-only UI host. Audio stays in the existing native playback engine."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread, Lock
from urllib.parse import urlsplit, unquote
import hashlib
import json
import os
import mimetypes
import time


def waveform(path, cache_directory):
    import numpy as np
    import soundfile as sf
    source = Path(path)
    stat = source.stat()
    key = hashlib.sha256(f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:v1".encode()).hexdigest()
    cache = Path(cache_directory) / (key + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    with sf.SoundFile(source) as audio:
        # A bounded envelope, computed once in a separate process, never per render.
        stride = max(1, int(np.ceil(len(audio) / 12000)))
        peaks = []
        for block in audio.blocks(blocksize=stride * 256, dtype="float32", always_2d=True):
            mono = np.max(np.abs(block), axis=1)
            peaks.extend(float(np.max(mono[i:i + stride])) for i in range(0, len(mono), stride))
        data = dict(duration=len(audio) / audio.samplerate, peaks=[round(v, 5) for v in peaks])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def analyze_import(path, directory):
    from ai_dj.pipeline.analyze import analyze_track
    from ai_dj.pipeline.cache import AnalysisCache
    cache = AnalysisCache(Path(directory) / ".ai_dj_cache")
    result = cache.get(Path(path)) or analyze_track(path)
    cache.put(Path(path), result)
    return result


class Workstation:
    def __init__(self, directory, sample_rate=44100, device=None, vocal_stems=True):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.engine = None
        self.status = "Analyzing library and planning the first ten tracks…"
        self.error = None
        self.imports = []
        self.waveforms = {}
        self.lock = Lock()
        self.workers = ProcessPoolExecutor(max_workers=1)
        self.background = ThreadPoolExecutor(max_workers=1)
        self.config = sample_rate, device, vocal_stems
        self.background.submit(self.initialize)

    def initialize(self):
        from ai_dj.playback.engine import PlaybackEngine
        engine = None
        try:
            sample_rate, device, vocal_stems = self.config
            engine = PlaybackEngine.from_library(self.directory, sample_rate=sample_rate, vocal_stems=vocal_stems)
            engine.start_output(device)
            self.engine = engine
            self.status = "Ready"
            self.error = None
        except Exception as exc:
            if engine:
                engine.close()
            self.error = str(exc)
            self.status = "Engine unavailable"

    def library(self):
        if not self.engine:
            return []
        return [dict(id=t.track_id, title=Path(t.source_path).stem,
                     filename=Path(t.source_path).name, duration=t.duration,
                     bpm=round(t.tempo.bpm, 1), key=t.key.key,
                     vocals=t.structure.vocal_activity.to_dict())
                for t in list(self.engine.tracks.values())]

    def state(self):
        engine = self.engine
        return dict(ready=engine is not None, status=self.status, error=self.error,
                    imports=list(self.imports), state=engine.state if engine else None,
                    audible=engine.audible_state if engine else None)

    def peaks(self, track_id):
        if not self.engine or track_id not in self.engine.tracks:
            raise ValueError("Unknown track")
        with self.lock:
            future = self.waveforms.get(track_id)
            if future is None:
                future = self.workers.submit(waveform, self.engine.tracks[track_id].source_path,
                                             str(self.directory / ".ai_dj_waveforms"))
                self.waveforms[track_id] = future
        return future.result() if future.done() else None

    def command(self, data):
        engine = self.engine
        if engine is None or engine.closed.is_set():
            raise ValueError("Audio engine is not ready")
        action = data.get("action")
        if action in ("play", "pause", "stop", "previous", "clear_overrides"):
            getattr(engine, action)()
        elif action == "next":
            engine.next(data.get("track"))
        elif action == "seek":
            engine.seek(float(data["seconds"]))
        elif action == "cue":
            engine.set_cue(data["track"], float(data["seconds"]))
        elif action == "point":
            seconds = float(data["seconds"])
            if data["track"] == engine.state["current_track"] and engine.state["playing"] and seconds < engine.state["position"] + .5:
                raise ValueError("Place the transition at least half a second ahead of playback")
            engine.set_transition_point(data["track"], seconds)
        elif action == "queue":
            engine.reorder_queue(data["tracks"])
        elif action == "volume":
            engine.set_volume(float(data["value"]))
        elif action == "auto":
            engine.set_auto_dj(data["enabled"])
        elif action == "transition":
            engine.transition_now(float(data.get("duration", 1)), data.get("track"))
        else:
            raise ValueError("Unknown control")

    def imported(self, path):
        item = dict(filename=path.name, status="Analyzing")
        self.imports.append(item)
        def finish():
            try:
                track = self.workers.submit(analyze_import, str(path), str(self.directory)).result()
                if self.engine:
                    self.engine.register_track(track)
                else:
                    self.initialize()
                item["status"] = "Ready"
            except Exception as exc:
                item["status"] = str(exc)
        self.background.submit(finish)

    def close(self):
        self.background.shutdown(wait=True)
        if self.engine:
            self.engine.close()
        self.workers.shutdown(wait=True, cancel_futures=True)


def make_server(app, port=8765):
    static = Path(__file__).with_name("static")
    allowed_origins = {origin.strip().rstrip('/') for origin in os.environ.get(
        "SYNC_FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(',') if origin.strip()}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, data, kind="application/json"):
            body = json.dumps(data, allow_nan=False).encode() if kind == "application/json" else data
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            # The API remains loopback-only by Host, while allowing the separately
            # deployed frontend to read state from the user's own engine.
            origin = self.headers.get("Origin")
            if origin and self.trusted():
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def trusted(self):
            expected = f"127.0.0.1:{self.server.server_port}"
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            return host in (expected, f"localhost:{self.server.server_port}") and (
                not origin or origin == f"http://{host}" or origin in allowed_origins)

        def do_GET(self):
            if not self.trusted():
                return self.send(403, {"error": "Local requests only"})
            path = urlsplit(self.path).path
            try:
                if path == "/api/state":
                    return self.send(200, app.state())
                if path == "/api/library":
                    return self.send(200, app.library())
                if path.startswith("/api/waveform/"):
                    peaks = app.peaks(unquote(path.rsplit("/", 1)[-1]))
                    return self.send(200 if peaks else 202, peaks or {"pending": True})
                name = "index.html" if path == "/" else path.lstrip("/")
                if name not in ("index.html", "app.js", "style.css"):
                    return self.send(404, {"error": "Not found"})
                return self.send(200, (static / name).read_bytes(),
                                 {"html": "text/html; charset=utf-8", "js": "text/javascript", "css": "text/css"}[name.rsplit(".", 1)[-1]])
            except Exception as exc:
                self.send(400, {"error": str(exc)})

        def do_OPTIONS(self):
            if not self.trusted():
                return self.send(403, {"error": "Local requests only"})
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin", "*"))
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_POST(self):
            if not self.trusted():
                return self.send(403, {"error": "Local requests only"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if self.path == "/api/upload":
                    name = unquote(self.headers.get("X-Filename", ""))
                    if Path(name).name != name or Path(name).suffix.lower() not in (".mp3", ".wav", ".flac"):
                        raise ValueError("Choose an MP3, WAV or FLAC file")
                    if not 0 < size <= 250 * 1024 * 1024:
                        raise ValueError("Files must be between 1 byte and 250 MB")
                    path = app.directory / name
                    # Exclusive creation avoids overwriting the user's music.
                    with path.open("xb") as output:
                        remaining = size
                        while remaining:
                            block = self.rfile.read(min(1024 * 1024, remaining))
                            if not block:
                                raise ValueError("Incomplete upload")
                            output.write(block)
                            remaining -= len(block)
                    app.imported(path)
                    return self.send(202, {"ok": True})
                if self.path != "/api/control" or not 0 < size < 65536:
                    raise ValueError("Invalid control request")
                app.command(json.loads(self.rfile.read(size)))
                self.send(200, {"ok": True})
            except Exception as exc:
                self.send(400, {"error": str(exc)})
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(directory="music", port=8765, sample_rate=44100, device=None, vocal_stems=True):
    app = Workstation(directory, sample_rate, device, vocal_stems)
    server = make_server(app, port)
    print(f"SYNC workstation: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()
