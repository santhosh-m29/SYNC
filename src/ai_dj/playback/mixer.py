"""Sample-clock two-deck mixer. Called by the producer, never by the device callback."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Deck:
    track_id: str
    audio: object
    frame: int = 0


class LiveMixer:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.current = None
        self.incoming = None
        self.event = None
        self.fade_frame = 0
        self.fade_length = 0
        self.completed = None
        self.last = np.zeros(2, dtype=np.float32)
        self.bridge = None
        self.bridge_frame = 0
        self.bridge_length = max(1, int(.02 * sample_rate))
        self.exhausted = False

    @property
    def position(self):
        return self.current.frame * self.current.audio.rate / self.sample_rate if self.current else 0.0

    def set_current(self, track_id, audio, position=0.0):
        self.bridge = self.last.copy()
        self.bridge_frame = 0
        frame = min(len(audio.samples) - 1, max(0, round(position / audio.rate * self.sample_rate)))
        self.current = Deck(track_id, audio, frame)
        self.cancel()
        self.exhausted = False

    def cancel(self):
        # Preserve continuity if a user cancels in the middle of a blend.
        if self.incoming is not None and self.fade_frame:
            self.bridge = self.last.copy()
            self.bridge_frame = 0
        self.incoming = self.event = None
        self.fade_frame = self.fade_length = 0

    def schedule(self, event, audio):
        self.event = event
        self.incoming = Deck(event.next_track, audio,
                             round(event.incoming_start_timestamp / audio.rate * self.sample_rate))
        self.fade_frame = 0
        self.fade_length = max(2, round(event.transition_duration * self.sample_rate))

    def render(self, frames):
        result = np.zeros((frames, 2), dtype=np.float32)
        written = 0
        self.completed = None
        while written < frames and self.current is not None:
            deck = self.current
            remaining = len(deck.audio.samples) - deck.frame
            if remaining <= 0:
                if self.incoming is not None:
                    self.completed = self.event
                    self.current = self.incoming
                    self.incoming = self.event = None
                    self.fade_frame = self.fade_length = 0
                    self.bridge = self.last.copy()
                    self.bridge_frame = 0
                    continue
                self.exhausted = True
                break
            until_event = (round(self.event.outgoing_transition_timestamp / deck.audio.rate * self.sample_rate)
                           - deck.frame) if self.event else remaining
            if self.event and until_event <= 0:
                count = min(frames - written, remaining, self.fade_length - self.fade_frame,
                            len(self.incoming.audio.samples) - self.incoming.frame)
                if count <= 0:
                    self.exhausted = True
                    break
                phase = (np.arange(count) + self.fade_frame) / (self.fade_length - 1) * (np.pi / 2)
                incoming = self.incoming
                result[written:written + count] = (
                    deck.audio.samples[deck.frame:deck.frame + count] * np.cos(phase)[:, None]
                    + incoming.audio.samples[incoming.frame:incoming.frame + count] * np.sin(phase)[:, None])
                deck.frame += count
                incoming.frame += count
                self.fade_frame += count
                if self.fade_frame == self.fade_length:
                    self.completed = self.event
                    self.current = incoming
                    self.incoming = self.event = None
                    self.fade_frame = self.fade_length = 0
            else:
                count = min(frames - written, remaining, max(1, until_event))
                result[written:written + count] = deck.audio.samples[deck.frame:deck.frame + count]
                deck.frame += count
            written += count
        if self.bridge is not None:
            count = min(frames, self.bridge_length - self.bridge_frame)
            phase = (np.arange(count) + self.bridge_frame + 1) / self.bridge_length
            result[:count] = self.bridge[None, :] * (1 - phase[:, None]) + result[:count] * phase[:, None]
            self.bridge_frame += count
            if self.bridge_frame >= self.bridge_length:
                self.bridge = None
        # Normalized deck peaks leave enough headroom even for fully correlated tracks.
        # A smooth saturator protects unexpected overs without a discontinuous hard clip.
        np.multiply(result, 1 / .98, out=result)
        np.tanh(result, out=result)
        result *= .98
        self.last[:] = result[-1]
        return result
