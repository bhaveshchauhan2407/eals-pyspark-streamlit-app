"""Tests for the train/test splits: src/split.py (offline) and src/online_split.py (online)."""

from collections import Counter

import pytest

from src.online_split import chronological_90_10_split
from src.split import leave_one_out_split

# A small history: user 1 has three items, user 2 has two, user 3 has only one
INTERACTIONS = [(1, 10), (1, 11), (1, 12), (2, 20), (2, 21), (3, 30)]


# ---------- Offline: leave-one-out ----------

def test_last_item_of_each_user_goes_to_test():
    _, test = leave_one_out_split(INTERACTIONS)
    assert sorted(test) == [(1, 12), (2, 21)]


def test_other_items_stay_in_train():
    train, _ = leave_one_out_split(INTERACTIONS)
    assert sorted(train) == [(1, 10), (1, 11), (2, 20), (3, 30)]


def test_user_with_single_interaction_is_train_only():
    train, test = leave_one_out_split(INTERACTIONS)
    assert (3, 30) in train
    assert all(u != 3 for u, _ in test)


def test_at_most_one_test_item_per_user():
    _, test = leave_one_out_split(INTERACTIONS)
    users_in_test = [u for u, _ in test]
    assert len(users_in_test) == len(set(users_in_test))


def test_no_interaction_in_both_train_and_test():
    train, test = leave_one_out_split(INTERACTIONS)
    assert set(train).isdisjoint(test)


def test_every_test_user_also_appears_in_train():
    # The model can only recommend for users it has seen during training
    train, test = leave_one_out_split(INTERACTIONS)
    train_users = {u for u, _ in train}
    assert all(u in train_users for u, _ in test)


def test_leave_one_out_loses_no_data():
    train, test = leave_one_out_split(INTERACTIONS)
    assert Counter(train + test) == Counter(INTERACTIONS)


def test_leave_one_out_empty_input():
    assert leave_one_out_split([]) == ([], [])


# ---------- Online: chronological 90/10 ----------

def make_events(n):
    """n events where event k is user k, item 100+k, at time k."""
    return [(k, 100 + k, k) for k in range(n)]


@pytest.mark.parametrize(
    "n, expected_train, expected_test",
    [(10, 9, 1), (20, 18, 2), (100, 90, 10), (5, 4, 1)],
)
def test_online_split_sizes(n, expected_train, expected_test):
    train, test = chronological_90_10_split(make_events(n))
    assert len(train) == expected_train
    assert len(test) == expected_test


def test_online_split_puts_newest_events_in_test():
    events = [(1, 10, 300), (2, 20, 100), (3, 30, 500), (4, 40, 200),
              (5, 50, 400), (6, 60, 600), (7, 70, 700), (8, 80, 800),
              (9, 90, 900), (10, 99, 1000)]
    _, test = chronological_90_10_split(events)
    assert test == [(10, 99)]


def test_online_split_train_is_in_time_order():
    events = [(3, 30, 300), (1, 10, 100), (2, 20, 200)]
    train, _ = chronological_90_10_split(events)
    # 3 events -> int(0.9 * 3) = 2 in train: the two oldest, oldest first
    assert train == [(1, 10), (2, 20)]


def test_online_split_removes_timestamps():
    train, test = chronological_90_10_split(make_events(10))
    assert all(len(pair) == 2 for pair in train + test)


def test_online_split_loses_no_data():
    events = make_events(25)
    train, test = chronological_90_10_split(events)
    assert sorted(train + test) == sorted((u, i) for u, i, _ in events)


def test_online_split_empty_input():
    assert chronological_90_10_split([]) == ([], [])
