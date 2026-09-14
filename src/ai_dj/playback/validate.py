"""Repeatable real-library validation; no transition WAV is generated.

python -m ai_dj.playback.validate --seconds 1800 --device-seconds 60
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from ai_dj.playback.engine import PlaybackEngine


def validate(directory, seconds, device_seconds=0, vocal_stems=False, device=None, sample_rate=44100):
    started = time.perf_counter()
    print("Preparing real library...", flush=True)
    with PlaybackEngine.from_library(directory, vocal_stems=vocal_stems, sample_rate=sample_rate) as engine:
        # Ensure the first test exercises a planned event, not just the deadline fallback.
        engine.future.result(timeout=300)
        engine.process()
        initial_event = engine.state["event"]
        print("Initial event prepared; running sample-clock validation...", flush=True)
        engine.play()
        peak = 0.0
        nonfinite = silent_transition_blocks = 0
        costs = []
        transition_blocks = 0
        transitions = []
        for _ in range(int(seconds * engine.sample_rate / engine.blocksize)):
            mixing = engine.mixer.event is not None and (
                engine.mixer.position + engine.blocksize / engine.sample_rate >=
                engine.mixer.event.outgoing_transition_timestamp)
            before = time.perf_counter()
            block = engine.process()
            costs.append(time.perf_counter() - before)
            peak = max(peak, float(np.max(np.abs(block))))
            nonfinite += int(not np.isfinite(block).all())
            if mixing:
                transition_blocks += 1
                silent_transition_blocks += int(np.sqrt(np.mean(block ** 2)) < 1e-6)
            if engine.state["transitions"] > len(transitions):
                transitions.append(engine.state["recent_events"][-1])
                # Simulated time must not unfairly outrun background preparation.
                # This wait is in the test driver, never the live playback path.
                engine.future.result(timeout=300)
        assert nonfinite == 0 and peak < .98
        if seconds:
            assert transitions, "Run was too short to exercise an automatic transition"
        assert silent_transition_blocks == 0, "Silent transition block detected"
        # Exercise controls on real decoded media using the same producer API.
        engine.seek(30)
        engine.process()
        seek_position = engine.state["position"]
        assert abs(seek_position - 30) < .1
        engine.pause()
        for _ in range(4):
            engine.process()
        paused_position = engine.state["position"]
        assert not engine.process().any()
        assert engine.state["position"] == paused_position
        engine.play()
        current = engine.state["current_track"]
        target = next((t for t in engine.tracks if t != current), current)
        engine.set_cue(target, 10)
        engine.reorder_queue(list(reversed(engine.queue.order)))
        engine.next(target)
        engine.process()
        if engine.manual_future:
            engine.manual_future[-1].result(timeout=120)
        for _ in range(20):
            engine.process()
        assert engine.state["current_track"] == target
        assert 10 <= engine.state["position"] < 11
        engine.previous()
        engine.process()
        if engine.manual_future:
            engine.manual_future[-1].result(timeout=120)
        for _ in range(20):
            engine.process()
        assert engine.state["current_track"] == current
        report = dict(library=[Path(t.source_path).name for t in engine.tracks.values()],
            simulated_seconds=seconds, peak=peak, nonfinite_blocks=nonfinite,
            silent_transition_blocks=silent_transition_blocks, transition_blocks=transition_blocks,
            transitions=transitions, initial_event=initial_event,
            producer_block_ms_p99=float(np.percentile(costs, 99) * 1000) if costs else None,
            producer_block_ms_max=max(costs) * 1000 if costs else None,
            controls="seek, pause/resume, cue, queue order, next, previous passed",
            vocal_states=engine.state["vocal_states"], errors=engine.state["errors"])
        if device_seconds:
            print("Starting hardware validation...", flush=True)
            engine.clear_overrides()
            engine.stop()
            for _ in range(4):
                engine.process()
            engine.future.result(timeout=300)
            engine.process()
            first_plan = engine.state["event"]
            if first_plan:
                engine.seek(max(0, first_plan["outgoing_transition_timestamp"] - 5))
                engine.process()
                engine.future.result(timeout=300)
                engine.process()
            engine.play()
            baseline = engine.transitions
            engine.start_output(device)
            deadline = time.monotonic() + device_seconds
            actions = [(device_seconds * .4, engine.pause),
                       (device_seconds * .45, engine.play),
                       (device_seconds * .6, engine.next),
                       (device_seconds * .8, engine.previous)]
            live_started = time.monotonic()
            while time.monotonic() < deadline and not engine.closed.is_set():
                elapsed = time.monotonic() - live_started
                while actions and elapsed >= actions[0][0]:
                    actions.pop(0)[1]()
                time.sleep(.05)
            report["device"] = dict(seconds=device_seconds, underruns=engine.ring.underruns,
                status_count=engine.device.status_count, transitions=engine.transitions - baseline,
                device_underflows=engine.device.underflows,
                state=engine.state)
            report["device"]["passed"] = (engine.ring.underruns == 0 and engine.device.underflows == 0
                                           and not engine.closed.is_set())
        report["elapsed_seconds"] = time.perf_counter() - started
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="music")
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--device-seconds", type=float, default=0)
    parser.add_argument("--device", type=int)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--vocal-stems", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("output/live-validation.json"))
    args = parser.parse_args()
    report = validate(args.directory, args.seconds, args.device_seconds, args.vocal_stems, args.device, args.sample_rate)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report.get("device", {}).get("passed") is False:
        raise SystemExit("Device validation failed; diagnostic report saved")


if __name__ == "__main__":
    main()
