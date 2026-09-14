from dataclasses import replace
from concurrent.futures import Future

import numpy as np
import pytest

from ai_dj.analysis.vocals import timeline_from_stems
from ai_dj.playback.audio import AudioLibrary, PreparedAudio, prepare_rate
from ai_dj.playback.device import AudioRing
from ai_dj.playback.engine import PlaybackEngine
from ai_dj.playback.mixer import LiveMixer
from ai_dj.playback.planner import PlaybackEvent, plan_event
from ai_dj.playback.queue import PlaybackQueue
from ai_dj.representation.structure import VocalActivity, VocalActivityEstimate
from tests.test_transition_planner import _track


def event(start=.5, duration=.25, rate=1):
    return PlaybackEvent(0, "a", "b", start, 0, duration, rate, 0, "test", 0, "unverified", "test")


def test_mixer_sample_accurate_and_independent_of_block_boundaries():
    audio = PreparedAudio(np.full((3000, 2), .2, dtype=np.float32))
    def run(blocks):
        mixer = LiveMixer(1000)
        mixer.set_current("a", audio)
        mixer.schedule(event(start=.513, duration=.267), audio)
        result = np.concatenate([mixer.render(n) for n in blocks])
        return result, mixer
    one, a = run([1500])
    split, b = run([113] * 13 + [31])
    np.testing.assert_allclose(one, split, atol=1e-7)
    assert a.current.track_id == b.current.track_id == "b"
    assert a.current.frame == b.current.frame == 987
    assert np.min(one[30:]) > .19
    assert np.max(np.abs(np.diff(one[30:, 0]))) < .002


def test_ring_wrap_and_underrun_telemetry():
    ring = AudioRing(8)
    assert ring.write(np.ones((6, 2)))
    out = np.empty((4, 2))
    ring.read_into(out)
    assert np.all(out == 1)
    assert ring.write(np.full((6, 2), 2))
    assert not ring.write(np.ones((1, 2)))
    out = np.empty((10, 2))
    ring.read_into(out)
    np.testing.assert_array_equal(out[:, 0], [1, 1, 2, 2, 2, 2, 2, 2, 0, 0])
    assert ring.underruns == 1


def test_queue_previous_walks_history_and_auto_visits_every_track():
    queue = PlaybackQueue(["a", "b", "c"], "a")
    queue.advance("b")
    assert queue.candidates() == ["c"]
    queue.advance("c")
    queue.advance(queue.history.pop(), manual=True, record_history=False)
    assert queue.current == "b"
    queue.advance(queue.history.pop(), manual=True, record_history=False)
    assert queue.current == "a"


def engine():
    tracks = [_track("a", duration=32), _track("b", duration=32)]
    audio = {t.track_id: PreparedAudio(np.full((32000, 2), .2, dtype=np.float32)) for t in tracks}
    return PlaybackEngine(tracks, audio, sample_rate=1000, blocksize=100)


def test_seek_invalidates_preparation_and_skip_continues():
    with engine() as e:
        e.play()
        e.process()
        old = e.generation
        e.seek(10)
        e.process()
        assert e.generation > old
        assert e.state["position"] == pytest.approx(10.1)
        if e.state["event"]:
            assert e.state["event"]["generation"] == e.generation
        e.next("b")
        samples = e.process()
        assert e.state["current_track"] == "b"
        assert np.min(samples) > 0
        assert e.state["selected_by"] == "manual"
        e.previous()
        e.process()
        assert e.state["current_track"] == "a"


def test_pause_stop_and_resume():
    with engine() as e:
        e.play()
        e.process()
        e.pause()
        e.process()
        pos = e.state["position"]
        assert not e.process().any()
        assert e.state["position"] == pos
        e.play()
        assert e.process().any()
        e.stop()
        e.process()
        assert e.state["position"] == 0
        assert not e.process().any()


def test_queue_and_manual_cue_survive_seek():
    with engine() as e:
        e.reorder_queue(["a", "b"])
        e.set_cue("b", 3)
        e.set_transition_point("a", 25)
        e.seek(12)
        e.process()
        assert e.points["a"] == 25 and e.cues["b"] == 3
        if e.future is not None:
            e.future.result(timeout=30)
            e.process()
        assert e.state["event"]["outgoing_transition_timestamp"] == 25
        assert e.state["event"]["incoming_start_timestamp"] == 3
        e.next("b")
        e.process()
        assert e.state["position"] == pytest.approx(3)
        with pytest.raises(ValueError):
            e.reorder_queue(["a", "a"])


def test_completed_old_generation_is_never_installed():
    with engine() as e:
        e._invalidate()
        stale = Future()
        stale.set_result((event(start=25), e.audio["b"]))
        e.future = stale
        e.process()
        assert e.mixer.event is None
        assert e.cancelled_preparations == 1


def test_new_vocal_evidence_updates_state_and_preserves_manual_overrides():
    from types import SimpleNamespace
    with engine() as e:
        e.set_transition_point("a", 25)
        e.set_cue("b", 3)
        e.process()
        before = e.generation
        result = [("a", VocalActivityEstimate((VocalActivity(0, 10, 1),), True, "stem", .5))]
        e.vocal_worker = SimpleNamespace(pending="b", process=SimpleNamespace(is_alive=lambda: True),
            poll=lambda: result.pop() if result else None, close=lambda: None)
        e.process()
        assert e.generation > before and e.state["vocal_states"]["a"] == "uncertain"
        assert e.points["a"] == 25 and e.cues["b"] == 3


def test_engine_preplans_lookahead_chain_before_play():
    tracks = [_track(str(i), duration=32) for i in range(12)]
    audio = {t.track_id: PreparedAudio(np.full((32000, 2), .2, dtype=np.float32)) for t in tracks}
    with PlaybackEngine(tracks, audio, sample_rate=1000, blocksize=100) as e:
        assert not e.playing
        assert len(e.lookahead) == 10
        assert e.state["preplanned_tracks"] == 10
        first = e.lookahead[e.queue.current]
        assert first.next_track != e.queue.current
        e.play()
        e.process()
        if e.future is not None:
            e.future.result(timeout=30)
            e.process()
        assert e.state["event"]["next_track"] == first.next_track


def test_lookahead_replenishment_is_async_after_five_completions():
    with engine() as e:
        e.completed_since_replan = 5
        e.mixer.completed = event()
        before = e.state["generation"]
        e.process()
        assert e.future is not None  # Replenishment shares the asynchronous plan/preparation job
        assert e.state["generation"] > before


def test_incoming_rate_and_playhead_survive_promotion():
    mixer = LiveMixer(1000)
    audio = PreparedAudio(np.full((3000, 2), .2, dtype=np.float32))
    fast = PreparedAudio(audio.samples, 1.05)
    mixer.set_current("a", audio)
    mixer.schedule(event(rate=1.05), fast)
    mixer.render(1000)
    assert mixer.current.audio.rate == 1.05
    assert mixer.position == pytest.approx(.5 * 1.05)


def test_lazy_library_is_bounded_and_keeps_current_native_audio(monkeypatch):
    tracks = [_track(str(i), duration=32) for i in range(10)]
    calls = []
    def decode(path, sr):
        calls.append(path)
        return PreparedAudio(np.ones((1000, 2), dtype=np.float32))
    monkeypatch.setattr("ai_dj.playback.audio.decode_track", decode)
    library = AudioLibrary(tracks, 1000, 8)
    library.current = ("0", library["0"])
    library.pinned = {"0"}
    for t in tracks[1:]:
        library[t.track_id]
    assert len(library.cache) <= 3 and len(calls) == 10
    assert library.resident("0") is not None
    assert library.resident("1") is None
    assert library["0"] is library.current[1]
    assert len(calls) == 10


def test_cold_manual_skip_prepares_without_blocking_or_losing_current_audio(monkeypatch):
    tracks = [_track(str(i), duration=32) for i in range(3)]
    library = AudioLibrary(tracks, 1000, 8)
    native = PreparedAudio(np.full((32000, 2), .2, dtype=np.float32))
    library.cache["0"] = library.cache["1"] = native
    monkeypatch.setattr("ai_dj.playback.engine.choose_next", lambda source, candidates: candidates[0])
    with PlaybackEngine(tracks, library, sample_rate=1000, blocksize=100) as e:
        e.future.result(timeout=30)
        e.process()
        e.play()
        e.process()
        pending = Future()
        original_submit = e.pool.submit
        monkeypatch.setattr(e.pool, "submit", lambda fn: pending)
        e.next("2")
        block = e.process()
        assert e.queue.current == "0" and np.min(block) > 0
        pending.set_result(native)
        monkeypatch.setattr(e.pool, "submit", original_submit)
        e.process()
        assert e.queue.current == "2"
        e.stop()
        e.process()
        assert e.state["position"] == 0


def test_end_seek_and_slow_preparation_keep_audio_continuous():
    with engine() as e:
        e.play()
        e.seek(1000)
        first = e.process()
        assert e.state["current_track"] == "b"
        assert np.isfinite(first).all() and np.max(first) > 0
        e._invalidate()
        e.future = Future()  # preparation has not finished
        e.mixer.current.frame = 31700
        blocks = [e.process() for _ in range(10)]
        assert np.min(np.concatenate(blocks)) > 0
        assert e.transitions >= 2


def test_planner_tempo_octaves_limits_and_manual_positions():
    a, b = _track("a", duration=32), _track("b", duration=32)
    a = replace(a, tempo=replace(a.tempo, bpm=120, confidence=1))
    b = replace(b, tempo=replace(b.tempo, bpm=60, confidence=1))
    p = plan_event(a, b, cue=3, transition=24)
    assert p.tempo_ratio == 1 and p.incoming_start_timestamp == 3 and p.outgoing_transition_timestamp == 24
    b = replace(b, tempo=replace(b.tempo, bpm=90))
    p = plan_event(a, b)
    assert p.tempo_ratio == 1 and p.transition_duration <= .25
    b = replace(b, tempo=replace(b.tempo, bpm=118))
    p = plan_event(a, b)
    assert p.tempo_ratio == pytest.approx(120 / 118)


def test_vocal_window_search_and_uncertainty():
    a, b = _track("a", duration=32), _track("b", duration=32)
    a = replace(a, structure=replace(a.structure, vocal_activity=VocalActivityEstimate(
        (VocalActivity(0, 24, 1),), True, "annotated", 1)))
    b = replace(b, structure=replace(b.structure, vocal_activity=VocalActivityEstimate(
        (VocalActivity(8, 32, 1),), True, "annotated", 1)))
    p = plan_event(a, b)
    assert p.vocal_safety == "safe"
    assert p.incoming_start_timestamp < 8
    uncertain = replace(a, structure=replace(a.structure, vocal_activity=replace(a.structure.vocal_activity, confidence=.5)))
    p = plan_event(uncertain, b)
    assert p.vocal_safety != "safe"


def test_stem_activity_includes_attack_and_release_without_claiming_calibration():
    voice, backing = np.zeros(5000), np.ones(5000) * .1
    voice[2000:3000] = .1
    timeline = timeline_from_stems(voice, backing, 1000)
    assert timeline.available and timeline.confidence < .6
    assert timeline.segments[0].start <= 1.8
    assert timeline.segments[0].end >= 3.2
    assert not timeline_from_stems(np.zeros(5000), backing, 1000).segments
    legacy = VocalActivityEstimate.from_dict({"available": True, "method": "unknown", "segments": []})
    assert legacy.confidence == 0


def test_pitch_preserving_preparation():
    sr = 8000
    wave = .2 * np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr)
    raw = PreparedAudio(np.repeat(wave.astype(np.float32)[:, None], 2, axis=1))
    stretched = prepare_rate(raw, 1.06)
    assert len(stretched.samples) == pytest.approx(len(raw.samples) / 1.06, abs=1)
    spectrum = np.abs(np.fft.rfft(stretched.samples[1000:9000, 0]))
    assert np.argmax(spectrum) == pytest.approx(440, abs=2)
