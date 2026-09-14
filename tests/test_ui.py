"""Behavioral checks for the workstation's real engine controls and waveform cache."""
from concurrent.futures import Future
from dataclasses import replace
import json
from threading import Thread
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
import pytest
import soundfile as sf

from ai_dj.ui.server import waveform, make_server
from tests.test_playback import engine


def settle(e):
    e.process()
    if e.future:
        e.future.result(timeout=30)
        e.process()


def test_cue_load_manual_schedule_and_auto_off():
    with engine() as e:
        e.set_auto_dj(False)
        e.set_cue("b", 7.25)
        settle(e)
        assert e.state["event"] is None
        e.next("b")
        e.process()
        assert e.state["position"] == pytest.approx(7.25)
        e.set_cue("a", 3.5)
        e.set_transition_point("b", 12)
        settle(e)
        event = e.state["event"]
        assert event["incoming_start_timestamp"] == 3.5
        assert event["outgoing_transition_timestamp"] == 12
        assert event["strategy"].startswith("manual_")
        e.play()
        count = int((12 - 7.25 + event["transition_duration"] + .5) * 10)
        blocks = [e.process() for _ in range(count)]
        assert e.state["current_track"] == "a"
        assert e.state["transitions"] == 1
        assert all(np.max(np.abs(block)) > 0 for block in blocks)


def test_queue_removal_volume_and_manual_natural_end():
    with engine() as e:
        e.reorder_queue(["a"])
        e.set_auto_dj(False)
        e.set_volume(.3)
        settle(e)
        assert e.state["queue_order"] == ["a"]
        e.play()
        e.process()
        block = e.process()
        assert np.max(block) == pytest.approx(.06, abs=.002)
        e.seek(31.8)
        for _ in range(5):
            e.process()
        assert not e.state["playing"]
        assert e.state["transitions"] == 0
        e.next("b")  # A library track can be loaded after removal from queue.
        e.process()
        assert "b" in e.state["queue_order"]


def test_cold_load_then_play_does_not_start_old_deck():
    with engine() as e:
        e.set_auto_dj(False)
        e.process()
        future = Future()
        e.manual_future = (e.generation, "b", .08, "next", future)
        e.play()
        assert not e.process().any()
        assert not e.playing
        future.set_result(e.audio["b"])
        e.process()
        assert e.state["current_track"] == "b"
        assert e.playing


def test_waveform_is_real_peak_envelope_and_reuses_disk_cache(tmp_path, monkeypatch):
    path = tmp_path / "wave.wav"
    samples = np.zeros((48000, 2), dtype=np.float32)
    samples[12000:24000] = .6
    sf.write(path, samples, 48000, subtype="FLOAT")
    result = waveform(path, tmp_path / "cache")
    assert result["duration"] == 1
    assert len(result["peaks"]) <= 12000
    assert max(result["peaks"][:3000]) == 0
    assert max(result["peaks"]) == pytest.approx(.6)
    monkeypatch.setattr(sf, "SoundFile", lambda *a, **k: pytest.fail("Decoded a cached waveform"))
    assert waveform(path, tmp_path / "cache") == result


def test_http_rejects_foreign_origin_and_dispatches_controls():
    class App:
        def state(self):
            return {"ready": True}
        def command(self, data):
            self.received = data
    app = App()
    server = make_server(app, 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert json.load(urlopen(base + "/api/state"))["ready"]
        payload = json.dumps({"action": "seek", "seconds": 12}).encode()
        urlopen(Request(base + "/api/control", data=payload)).close()
        assert app.received["seconds"] == 12
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/control", data=payload, headers={"Origin": "https://example.com"}))
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_upload_keeps_audio_bytes_and_requests_analysis(tmp_path):
    class App:
        directory = tmp_path
        imported_path = None
        def imported(self, path):
            self.imported_path = path
    app = App()
    server = make_server(app, 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = tmp_path / "source.wav"
    sf.write(source, np.sin(np.arange(48000) * .05), 48000)
    payload = source.read_bytes()
    url = f"http://127.0.0.1:{server.server_port}/api/upload"
    try:
        with urlopen(Request(url, data=payload, headers={"X-Filename": "import.wav"})) as response:
            assert response.status == 202
        assert app.imported_path.read_bytes() == payload
        # Import must never replace an existing song.
        with pytest.raises(HTTPError):
            urlopen(Request(url, data=b"replace", headers={"X-Filename": "import.wav"}))
        assert app.imported_path.read_bytes() == payload
        with pytest.raises(HTTPError):
            urlopen(Request(url, data=payload, headers={"X-Filename": "../escape.wav"}))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_imported_analysis_registers_with_running_audio_library(tmp_path):
    from ai_dj.playback.audio import AudioLibrary
    from ai_dj.playback.engine import PlaybackEngine
    from tests.test_transition_planner import _track
    analyzed = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.wav"
        sf.write(path, np.sin(np.arange(32000) * .2) * .2, 1000)
        analyzed.append(replace(_track(name, duration=32), source_path=str(path)))
    audio = AudioLibrary(analyzed[:1], 1000, 16)
    with PlaybackEngine(analyzed[:1], audio, sample_rate=1000, blocksize=100) as e:
        e.set_auto_dj(False)
        e.play()
        e.process()
        e.register_track(analyzed[1])
        e.process()
        assert "b" in e.tracks and "b" in audio.tracks
        assert e.state["playing"]
        e.next("b")
        e.process()
        if e.manual_future:
            e.manual_future[-1].result(timeout=30)
        e.process()
        assert e.state["current_track"] == "b"
