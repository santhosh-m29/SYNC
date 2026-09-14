"""Queue policy, independent of device and DSP state."""
from dataclasses import dataclass, field


@dataclass
class PlaybackQueue:
    order: list[str]
    current: str
    manual_order: bool = False
    history: list[str] = field(default_factory=list)
    selected_by: str = "automatic"
    played: set[str] = field(default_factory=set)

    def __post_init__(self):
        self.reorder(self.order, manual=False)
        if self.current not in self.order:
            raise ValueError("Current track must belong to queue")
        self.played.add(self.current)

    def reorder(self, order, *, manual=True):
        order = list(order)
        if not order or len(set(order)) != len(order):
            raise ValueError("Queue must contain unique track IDs")
        if self.order and set(order) != set(self.order):
            raise ValueError("Reorder must contain every queued track exactly once")
        self.order = order
        self.manual_order = manual

    def candidates(self):
        index = self.order.index(self.current)
        rotated = self.order[index + 1:] + self.order[:index]
        if not rotated:
            return [self.current]
        if self.manual_order:
            return rotated[:1]
        unplayed = [item for item in rotated if item not in self.played]
        return unplayed or rotated

    def advance(self, track_id, *, manual=False, record_history=True):
        if track_id not in self.order:
            raise ValueError("Unknown queue track")
        if record_history:
            self.history.append(self.current)
            self.history = self.history[-100:]
        if len(self.played) == len(self.order):
            self.played.clear()
        self.current = track_id
        self.played.add(track_id)
        self.selected_by = "manual" if manual else "automatic"
