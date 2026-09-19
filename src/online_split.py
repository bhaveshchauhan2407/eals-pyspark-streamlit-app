from typing import List, Tuple


def chronological_90_10_split(events: List[Tuple[int, int, int]]):
    """
    events: list of (user_id, item_id, timestamp), already sorted by timestamp
    train = first 90%
    test = last 10%
    """
    events_sorted = sorted(events, key=lambda x: x[2])

    cutoff = int(0.9 * len(events_sorted))
    train_events = events_sorted[:cutoff]
    test_events = events_sorted[cutoff:]

    train_interactions = [(u, i) for u, i, _ in train_events]
    test_interactions = [(u, i) for u, i, _ in test_events]

    return train_interactions, test_interactions