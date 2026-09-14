"""Control API, preparation worker, sample-clock scheduler, and live producer.

All mutable mixer/queue state belongs to the producer thread. Public controls
enqueue commands. The device callback only consumes a bounded PCM ring.
"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from queue import SimpleQueue, Empty
from threading import Thread, Event
import time
import logging
from collections import deque


def _plan_chain(tracks, order, current, manual, cues, points, generation, position=0, rate=1):
    planned, seen = {}, {current}
    source = tracks[current]
    rotated = order[order.index(current)+1:] + order[:order.index(current)]
    for _ in range(min(10, len(rotated))):
        candidates = [tracks[t] for t in rotated if t not in seen]
        if not candidates:
            break
        destination = candidates[0] if manual else choose_next(source, candidates)
        event = plan_event(source, destination, position=position, source_rate=rate,
                           generation=generation, cue=cues.get(destination.track_id),
                           transition=points.get(source.track_id))
        if source.track_id in points or destination.track_id in cues:
            event = replace(event, strategy="manual_" + event.strategy)
        planned[source.track_id] = event
        seen.add(destination.track_id)
        source = destination
        position = event.incoming_start_timestamp + event.transition_duration * event.tempo_ratio
        rate = event.tempo_ratio
    return planned

import numpy as np

from ai_dj.pipeline.analyze import analyze_library
from ai_dj.pipeline.cache import AnalysisCache
from ai_dj.playback.audio import AudioLibrary, PreparedAudio, prepare_rate
from ai_dj.playback.device import AudioRing, OutputDevice
from ai_dj.playback.mixer import LiveMixer
from ai_dj.playback.planner import plan_event, choose_next, PlaybackEvent
from ai_dj.playback.queue import PlaybackQueue

LOGGER = logging.getLogger(__name__)


class PlaybackEngine:
    def __init__(self, analyses, audio, *, sample_rate=44100, blocksize=512):
        if not analyses or sample_rate <= 0 or blocksize <= 0:
            raise ValueError("A nonempty analyzed library and positive audio configuration are required")
        self.tracks = {t.track_id: t for t in analyses}
        if set(audio) != set(self.tracks):
            raise ValueError("Every track needs prepared audio")
        self.audio = audio
        self.sample_rate, self.blocksize = sample_rate, blocksize
        first = analyses[0].track_id
        self.queue = PlaybackQueue(list(self.tracks), first)
        self.mixer = LiveMixer(sample_rate)
        self.mixer.set_current(first, audio[first])
        self.commands = SimpleQueue()
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sync-prepare")
        # Candidate search is Python-heavy: a thread would compete with the
        # PortAudio callback for the GIL. Run it in a separate process.
        self.plan_pool = ProcessPoolExecutor(max_workers=1)
        self.future = None
        self.future_target = None
        self.plan_hint = None
        self.failed_tracks = set()
        self.manual_future = None
        self.vocal_worker = None
        self.vocal_remaining = []
        self.lookahead_limit = 10
        self.lookahead = {}
        self.lookahead_future = None
        self.completed_since_replan = 0
        self.generation = 0
        self.cues, self.points = {}, {}
        self.playing = False
        self.auto_dj = True
        self.volume = self.output_volume = 1.0
        self.clock_history = deque(maxlen=2048)
        self.master = 0.0
        self.pending_stop = False
        self.pending_play = False
        self.transitions = 0
        self.cancelled_preparations = 0
        self.errors = []
        self.recent_events = []
        # Windows MME can request a burst of several callbacks for its hardware
        # buffer. Keep 350 ms ready, not merely one callback's worth of samples.
        self.buffer_blocks = max(8, int(np.ceil(.35 * sample_rate / blocksize)))
        self.ring = AudioRing(blocksize * self.buffer_blocks * 2)
        self.maximum_producer_seconds = 0.0
        self.device = None
        self.thread = None
        self.closed = Event()
        self._snapshot = {}
        self._build_lookahead()
        self._request_plan()
        self._publish()

    @classmethod
    def from_library(cls, directory="music", *, sample_rate=44100, memory_limit_mb=1024,
                     vocal_stems=True):
        root = Path(directory)
        result = analyze_library(root, AnalysisCache(root / ".ai_dj_cache"))
        if not result.analyses:
            raise ValueError("No decodable songs in music library")
        tracks = result.analyses
        if vocal_stems:
            from ai_dj.analysis.vocals import read_cached_track_vocals
            tracks = [replace(t, structure=replace(t.structure, vocal_activity=(
                read_cached_track_vocals(t.source_path, root / ".ai_dj_stems") or t.structure.vocal_activity))) for t in tracks]
        audio = AudioLibrary(tracks, sample_rate, memory_limit_mb)
        first = tracks[0]
        target = choose_next(first, tracks[1:] or tracks)
        audio.pinned = {first.track_id, target.track_id}
        audio[first.track_id]
        audio[target.track_id]
        audio.current = (first.track_id, audio[first.track_id])
        engine = cls(tracks, audio, sample_rate=sample_rate)
        engine.errors.extend(f"{path}: {error}" for path, error in result.failures)
        if vocal_stems:
            import importlib.util
            if importlib.util.find_spec("demucs") is not None:
                from ai_dj.playback.vocal_worker import VocalWorker
                engine.vocal_worker = VocalWorker(root / ".ai_dj_stems")
                engine.vocal_remaining = [first.track_id, target.track_id] + [
                    t.track_id for t in tracks if t.track_id not in (first.track_id, target.track_id)]
                engine.vocal_remaining = list(dict.fromkeys(engine.vocal_remaining))
                engine.vocal_remaining = [t for t in engine.vocal_remaining if not engine.tracks[t].structure.vocal_activity.available]
            else:
                engine.errors.append("Optional Demucs unavailable; vocal detection remains unavailable")
        return engine

    def _poll_vocals(self):
        if self.vocal_worker is None:
            return
        worker = self.vocal_worker
        result = worker.poll()
        if result:
            track_id, timeline = result
            track = self.tracks[track_id]
            self.tracks[track_id] = replace(track, structure=replace(track.structure, vocal_activity=timeline))
            event = self.mixer.event
            relevant = track_id in (self.queue.current, self._destination()) or (event and track_id == event.next_track)
            if relevant and not self.mixer.fade_frame and not self.manual_future and not (
                    event and event.strategy.startswith("manual_")):
                self._invalidate()
                self._request_plan()
        if not worker.process.is_alive():
            self.errors.append("Vocal worker stopped; remaining detections unavailable")
            worker.close()
            self.vocal_worker = None
        elif worker.pending is None and self.vocal_remaining:
            preferred = self._destination()
            target = preferred if preferred in self.vocal_remaining and self.queue.current not in self.vocal_remaining else self.vocal_remaining[0]
            self.vocal_remaining.remove(target)
            worker.submit(self.tracks[target])

    def play(self):
        self.commands.put(("play", None))

    def pause(self):
        self.commands.put(("pause", None))

    def stop(self):
        self.commands.put(("stop", None))

    def seek(self, seconds):
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Seek must be a finite nonnegative timestamp")
        self.commands.put(("seek", float(seconds)))

    def next(self, track_id=None):
        if track_id is not None and track_id not in self.tracks:
            raise ValueError("Unknown track")
        self.commands.put(("next", track_id))

    def previous(self):
        self.commands.put(("previous", None))

    def set_cue(self, track_id, seconds):
        self._validate_time(track_id, seconds)
        self.commands.put(("cue", (track_id, float(seconds))))

    def set_transition_point(self, track_id, seconds):
        self._validate_time(track_id, seconds)
        self.commands.put(("point", (track_id, float(seconds))))

    def clear_overrides(self):
        self.commands.put(("clear", None))

    def reorder_queue(self, track_ids):
        if not track_ids or len(set(track_ids)) != len(track_ids) or not set(track_ids) <= set(self.tracks):
            raise ValueError("Provide unique known track IDs")
        if self.state["current_track"] not in track_ids:
            raise ValueError("Keep the current track in the queue")
        self.commands.put(("order", list(track_ids)))

    def set_volume(self, value):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Volume must be between zero and one")
        self.commands.put(("volume", float(value)))

    def set_auto_dj(self, enabled):
        self.commands.put(("auto", bool(enabled)))

    def register_track(self, track):
        self.commands.put(("register", track))

    def transition_now(self, duration=1.0, track_id=None):
        if not np.isfinite(duration) or duration < .02 or duration > 30:
            raise ValueError("Manual transition duration must be 0.02–30 seconds")
        if track_id is not None and track_id not in self.tracks:
            raise ValueError("Unknown track")
        self.commands.put(("transition", (track_id, duration)))

    @property
    def state(self):
        # The published dictionary is replaced atomically and never mutated.
        import copy
        return copy.deepcopy(self._snapshot)

    def _validate_time(self, track_id, seconds):
        if track_id not in self.tracks or not np.isfinite(seconds) or not 0 <= seconds < self.tracks[track_id].duration:
            raise ValueError("Timestamp must lie within a known track")

    def _invalidate(self):
        self.generation += 1
        if self.mixer.event:
            self.plan_hint = self.mixer.event.next_track
        self.mixer.cancel()
        if self.future:
            self.future.cancel()
        self.future = None
        if self.manual_future:
            self.manual_future[-1].cancel()
            self.manual_future = None

    def _resident(self, track_id):
        return self.audio.resident(track_id) if isinstance(self.audio, AudioLibrary) else self.audio.get(track_id)

    def _destination(self):
        planned = self.lookahead.get(self.queue.current)
        candidates = [t for t in self.queue.candidates() if t not in self.failed_tracks]
        if planned and planned.next_track in candidates:
            return planned.next_track
        return candidates[0] if candidates else self.queue.current

    def _build_lookahead(self):
        self.lookahead = _plan_chain(self.tracks, self.queue.order, self.queue.current,
            self.queue.manual_order, self.cues, self.points, self.generation,
            self.mixer.position, self.mixer.current.audio.rate)
        self.completed_since_replan = 0

    def _request_plan(self):
        if isinstance(self.audio, AudioLibrary):
            prepared = self.mixer.current.audio
            native = prepared.original_samples if prepared.original_samples is not None else prepared.samples
            self.audio.current = (self.queue.current, PreparedAudio(native))
            self.audio.pinned = {self.queue.current}
        if not self.auto_dj and self.queue.current not in self.points:
            return
        source = self.tracks[self.queue.current]
        generation, position = self.generation, self.mixer.position
        rate = self.mixer.current.audio.rate
        tracks, order = dict(self.tracks), list(self.queue.order)
        manual, cues, points = self.queue.manual_order, dict(self.cues), dict(self.points)
        cached = dict(self.lookahead)
        planned = cached.get(source.track_id)
        if planned and (planned.next_track not in order or position > planned.outgoing_transition_timestamp):
            planned = None
        hint = self.plan_hint if self.plan_hint in self.queue.candidates() else None
        self.plan_hint = None
        def work():
            if hint and not planned:
                event = self.plan_pool.submit(plan_event, source, tracks[hint], position, rate,
                    generation, cues.get(hint), points.get(source.track_id)).result()
                if source.track_id in points or hint in cues:
                    event = replace(event, strategy="manual_" + event.strategy)
                chain = dict(cached)
                chain[source.track_id] = event
            else:
                chain = cached if planned else self.plan_pool.submit(_plan_chain, tracks, order,
                source.track_id, manual, cues, points, generation, position, rate).result()
            event = chain.get(source.track_id)
            if event is None:
                return None, None, chain
            event = replace(event, generation=generation)
            native = self.audio[event.next_track]
            return event, prepare_rate(native, event.tempo_ratio), chain
        self.future_target = planned.next_track if planned else None
        self.future = self.pool.submit(work)

    def _fallback(self, target=None, duration=.05, immediate=False, prepared=None):
        target = target or self._destination()
        native = prepared if prepared is not None else self._resident(target)
        if native is None:
            # A slow/newly changed queue must not force decoding in the producer.
            for candidate in [t for t in self.queue.order if t != self.queue.current] + [self.queue.current]:
                native = self._resident(candidate)
                if native is not None:
                    target = candidate
                    break
        source = self.mixer.current
        remaining = (len(source.audio.samples) - source.frame) / self.sample_rate
        entry = self.cues.get(target, 0.0)
        seconds = min(duration, remaining, (len(native.samples) / self.sample_rate - entry))
        seconds = max(2 / self.sample_rate, seconds)
        start = self.mixer.position if immediate else max(self.mixer.position,
            self.tracks[source.track_id].duration - seconds * source.audio.rate)
        return PlaybackEvent(self.generation, source.track_id, target, start, entry, seconds,
                             1.0, 0.0, "manual_crossfade" if immediate else "preparation_deadline_handoff",
                             0.0, "unverified", "Resident native-rate fallback; vocal safety unverified"), native

    def _commands(self):
        # Bound command handling so an unbounded producer cannot starve audio.
        for _ in range(64):
            try:
                command, value = self.commands.get_nowait()
            except Empty:
                break
            if command == "play":
                if self.manual_future and not self.playing:
                    self.pending_play = True
                else:
                    self.playing = True
                self.pending_stop = False
            elif command == "volume":
                self.volume = value
            elif command == "register":
                self.tracks[value.track_id] = value
                if isinstance(self.audio, AudioLibrary):
                    self.audio.tracks[value.track_id] = value
            elif command == "pause":
                self.pending_play = False
                self.playing = False
            elif command == "stop":
                self.pending_play = False
                self.playing = False
                self.pending_stop = True
                self._invalidate()
            elif command == "seek":
                self._invalidate()
                value = min(value, self.tracks[self.queue.current].duration)
                self.mixer.set_current(self.queue.current, self.mixer.current.audio, value)
                self.lookahead.clear()
                self._request_plan()
            elif command in ("next", "previous", "transition"):
                self._invalidate()
                duration = .08
                if command == "previous":
                    target = self.queue.history.pop() if self.queue.history else self.queue.current
                elif command == "transition":
                    target, duration = value
                    target = target or self._destination()
                else:
                    target = value or self._destination()
                if target not in self.queue.order:
                    self.queue.order.append(target)
                self.failed_tracks.discard(target)
                ready = self._resident(target)
                if ready is None:
                    generation = self.generation
                    self.manual_future = (generation, target, duration, command,
                                          self.pool.submit(lambda t=target: self.audio[t]))
                    continue
                if self.playing:
                    event, ready = self._fallback(target, duration, immediate=True, prepared=ready)
                    if command == "previous":
                        event = replace(event, strategy="manual_previous")
                    self.mixer.schedule(event, ready)
                else:
                    self.queue.advance(target, manual=True, record_history=command != "previous")
                    self.mixer.set_current(target, ready, self.cues.get(target, 0))
                    self._request_plan()
            else:
                self._invalidate()
                if command == "cue":
                    self.cues[value[0]] = value[1]
                    if value[0] == self.queue.current and not self.playing:
                        self.mixer.set_current(value[0], self.mixer.current.audio, value[1])
                elif command == "point":
                    self.points[value[0]] = value[1]
                elif command == "clear":
                    self.cues.clear()
                    self.points.clear()
                elif command == "order":
                    self.queue.order = value
                    self.queue.manual_order = True
                elif command == "auto":
                    self.auto_dj = value
                self.lookahead.clear()
                self._request_plan()

    def process(self, frames=None):
        """Produce one bounded block. For headless tests, call from one thread only."""
        frames = frames or self.blocksize
        self._commands()
        self._poll_vocals()
        if self.lookahead_future is not None and self.lookahead_future.done():
            try:
                self.lookahead_future.result()
            except Exception as error:
                self.errors.append(f"Lookahead planning failed: {error}")
            self.lookahead_future = None
        if self.manual_future and self.manual_future[-1].done():
            generation, target, duration, command, future = self.manual_future
            self.manual_future = None
            try:
                audio = future.result()
                if generation == self.generation:
                    if self.playing:
                        event, audio = self._fallback(target, duration, immediate=True, prepared=audio)
                        if command == "previous":
                            event = replace(event, strategy="manual_previous")
                        self.mixer.schedule(event, audio)
                    else:
                        self.queue.advance(target, manual=True, record_history=command != "previous")
                        self.mixer.set_current(target, audio, self.cues.get(target, 0))
                        if self.pending_play:
                            self.pending_play = False
                            self.playing = True
                        self._request_plan()
            except Exception as error:
                self.errors.append(str(error))
                self.failed_tracks.add(target)
                self._request_plan()
        if self.future and self.future.done():
            future, self.future = self.future, None
            try:
                result = future.result()
                event, audio = result[:2]
                if len(result) == 3 and event and event.generation == self.generation:
                    self.lookahead = result[2]
                if (event and event.generation == self.generation and not self.mixer.fade_frame
                        and event.outgoing_transition_timestamp >= self.mixer.position):
                    self.mixer.schedule(event, audio)
                    LOGGER.info(
                        "[DJ] Exit: %s @ %05.1fs | Next: %s @ %05.1fs | Transition score: %.0f | Transition scheduled",
                        event.current_track, event.outgoing_transition_timestamp,
                        event.next_track, event.incoming_start_timestamp,
                        event.transition_score * 100,
                    )
                else:
                    self.cancelled_preparations += 1
            except Exception as error:
                self.errors.append(str(error))
                self.errors = self.errors[-20:]
                if self.future_target and self.future_target != self.queue.current:
                    self.failed_tracks.add(self.future_target)
                    self._request_plan()
        # A ready native-rate deck exists even if analysis/tempo preparation misses its deadline.
        remaining = (len(self.mixer.current.audio.samples) - self.mixer.current.frame) / self.sample_rate
        if self.auto_dj and len(self.queue.order) > 1 and not self.mixer.event and remaining <= max(2.0, frames / self.sample_rate * 2):
            event, ready = self._fallback(duration=min(.25, remaining / 2))
            self.mixer.schedule(event, ready)
        if self.playing or self.master > 0:
            output = self.mixer.render(frames)
        else:
            output = np.zeros((frames, 2), dtype=np.float32)
        if self.mixer.exhausted and not self.mixer.event:
            self.playing = False
        volume = np.linspace(self.output_volume, self.volume, frames, dtype=np.float32)
        output *= volume[:, None]
        self.output_volume = self.volume
        target = 1.0 if self.playing else 0.0
        if self.master != target:
            step = np.arange(1, frames + 1) / (.02 * self.sample_rate)
            envelope = np.clip(self.master + (step if target else -step), 0, 1)
            output *= envelope[:, None]
            self.master = float(envelope[-1])
        if self.pending_stop and self.master == 0:
            self.pending_stop = False
            self.mixer.set_current(self.queue.current, self._resident(self.queue.current))
            self._request_plan()
        if self.mixer.completed:
            event = self.mixer.completed
            self.mixer.completed = None
            self.queue.advance(event.next_track, manual=event.strategy.startswith("manual_"),
                               record_history=event.strategy != "manual_previous")
            if isinstance(self.audio, AudioLibrary):
                self.audio.pinned = {event.next_track}
                prepared = self.mixer.current.audio
                native = prepared.original_samples if prepared.original_samples is not None else prepared.samples
                self.audio.current = (event.next_track, PreparedAudio(native))
            self.transitions += 1
            self.completed_since_replan += 1
            self.recent_events = (self.recent_events + [asdict(event)])[-20:]
            self.generation += 1
            if self.completed_since_replan >= 5:
                self.lookahead.clear()
                self.completed_since_replan = 0
            if self.manual_future:
                self.manual_future = (self.generation, *self.manual_future[1:])
            else:
                self._request_plan()
        self._publish()
        return output

    def _publish(self):
        self._snapshot = dict(current_track=self.queue.current, position=self.mixer.position,
            playing=self.playing, auto_dj=self.auto_dj, volume=self.volume,
            cues=dict(self.cues), points=dict(self.points),
            plans=[asdict(e) for e in self.lookahead.values()],
            incoming_position=(self.mixer.incoming.frame * self.mixer.incoming.audio.rate / self.sample_rate
                               if self.mixer.incoming and self.mixer.fade_frame else None),
            queue_manual_order=self.queue.manual_order, queue_order=list(self.queue.order), upcoming=self.queue.candidates(),
            selected_by=self.queue.selected_by, transition_state="mixing" if self.mixer.fade_frame else
            "scheduled" if self.mixer.event else "preparing", generation=self.generation,
            event=asdict(self.mixer.event) if self.mixer.event else None,
            tempo_ratio=self.mixer.current.audio.rate, transitions=self.transitions,
            recent_events=list(self.recent_events), errors=list(self.errors),
            failed_tracks=sorted(self.failed_tracks),
            underruns=self.ring.underruns, device_status_count=self.device.status_count if self.device else 0,
            device_underflows=self.device.underflows if self.device else 0,
            buffered_seconds=self.ring.available / self.sample_rate,
            maximum_producer_seconds=self.maximum_producer_seconds,
            lookahead_replan_pending=self.lookahead_future is not None,
            vocal_analysis_pending=(self.vocal_worker.pending if self.vocal_worker else None),
            preplanned_tracks=len(self.lookahead),
            completed_since_replan=self.completed_since_replan,
            vocal_states={k: ("unavailable" if not t.structure.vocal_activity.available else
                "confident" if t.structure.vocal_activity.confidence >= .6 else "uncertain") for k, t in self.tracks.items()})

    @property
    def audible_state(self):
        # Select the state of the block reaching the DAC, including deck changes.
        latency = self.device.stream.latency if self.device else 0
        frame = self.ring.read_frame - int(latency * self.sample_rate)
        history = list(self.clock_history)
        selected = next((state for end, state in reversed(history) if end <= frame),
                        history[0][1] if history else self.state)
        return {key: selected.get(key) for key in ("current_track", "position", "playing",
            "event", "incoming_position", "transition_state", "tempo_ratio")}

    def start_output(self, device=None):
        if self.thread is not None:
            raise RuntimeError("Output already started")
        self.device = OutputDevice(self.ring, self.sample_rate, self.blocksize, device)
        # Prefill before opening the stream; only bounded blocks, not transition files.
        for _ in range(self.buffer_blocks):
            self.ring.write(self.process())
            self.clock_history.append((self.ring.write_frame, self._snapshot))
        self.thread = Thread(target=self._produce, name="sync-mix", daemon=True)
        self.thread.start()
        self.device.start()

    def _produce(self):
        try:
            while not self.closed.is_set():
                if self.ring.available <= self.blocksize * (self.buffer_blocks - 1):
                    started = time.perf_counter()
                    self.ring.write(self.process())
                    self.clock_history.append((self.ring.write_frame, self._snapshot))
                    self.maximum_producer_seconds = max(self.maximum_producer_seconds, time.perf_counter() - started)
                else:
                    self.closed.wait(.001)
        except Exception as error:
            self.errors.append(f"Producer failed: {error}")
            self._publish()
            self.closed.set()

    def close(self):
        if self.device and self.thread and self.thread.is_alive() and not self.closed.is_set():
            self.stop()
            deadline = time.monotonic() + 2.0
            while self.state["playing"] and time.monotonic() < deadline:
                time.sleep(.01)
            # Let the short stop envelope and already queued device audio drain.
            time.sleep(min(1.0, .05 + self.ring.available / self.sample_rate + self.device.stream.latency))
        self.closed.set()
        if self.thread:
            self.thread.join(timeout=2)
        if self.device:
            self.device.close()
        if self.vocal_worker:
            self.vocal_worker.close()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.plan_pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
