"""Convert clip-level predictions into reviewable potential-incident events."""
from dataclasses import dataclass, asdict

@dataclass
class Event:
    activity: str
    confidence: float
    start_time: float
    end_time: float
    duration: float
    status: str = "Potential Incident"


def generate_events(predictions, confidence_threshold=0.70, min_consecutive=2):
    """Merge consecutive high-confidence same-label predictions.

    predictions: iterable of dicts with start_time, end_time, activity, confidence.
    """
    events = []
    current = None
    streak = 0
    for p in predictions:
        active = p["confidence"] >= confidence_threshold
        same = current is not None and p["activity"] == current["activity"]
        if active and (same or current is None):
            if current is None:
                current = {"activity": p["activity"], "start_time": p["start_time"],
                           "end_time": p["end_time"], "peak": p["confidence"]}
            else:
                current["end_time"] = p["end_time"]
                current["peak"] = max(current["peak"], p["confidence"])
            streak += 1
        else:
            if current is not None and streak >= min_consecutive:
                events.append(Event(current["activity"], current["peak"], current["start_time"],
                                    current["end_time"], current["end_time"] - current["start_time"]))
            current = None
            streak = 0
    if current is not None and streak >= min_consecutive:
        events.append(Event(current["activity"], current["peak"], current["start_time"],
                            current["end_time"], current["end_time"] - current["start_time"]))
    return [asdict(e) for e in events]
