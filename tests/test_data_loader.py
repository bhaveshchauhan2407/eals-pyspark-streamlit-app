"""Tests for src/data_loader.py: reading, cleaning and filtering the ratings file."""

import pandas as pd
import pytest

from src.data_loader import (
    dataframe_to_events,
    dataframe_to_interactions,
    get_unique_users_items,
    load_events,
    load_interactions,
    read_raw_ratings,
)


# ---------- Reading the raw file ----------

def test_read_raw_handles_tabs_and_multiple_spaces(ratings_file):
    path = ratings_file(["1\t10\t4.0\t100", "2    20  5.0   200"])
    df = read_raw_ratings(path)
    assert list(df.columns) == ["user_id", "item_id", "rating", "timestamp"]
    assert df.shape == (2, 4)
    assert df.loc[1, "item_id"] == 20


def test_read_raw_turns_non_numbers_into_nan(ratings_file):
    path = ratings_file(["abc 10 4.0 100", "2 20 5.0 200"])
    df = read_raw_ratings(path)
    assert pd.isna(df.loc[0, "user_id"])


def test_read_raw_skips_lines_with_too_many_fields(ratings_file):
    path = ratings_file(["1 10 4.0 100", "2 20 5.0 200 999", "3 30 3.0 300"])
    df = read_raw_ratings(path)
    assert df["user_id"].tolist() == [1, 3]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_interactions(str(tmp_path / "does_not_exist.txt"))


@pytest.mark.parametrize("loader", [load_interactions, load_events])
def test_empty_file_gives_empty_result(ratings_file, loader):
    df = loader(ratings_file([]))
    assert df.empty


@pytest.mark.parametrize("loader", [load_interactions, load_events])
def test_header_line_is_ignored(ratings_file, loader):
    path = ratings_file(["user_id item_id rating timestamp", "1 10 4.0 100"])
    df = loader(path)
    assert len(df) == 1
    assert df.loc[0, "user_id"] == 1


# ---------- Offline loader: load_interactions ----------

def test_load_interactions_keeps_only_user_and_item(ratings_file):
    df = load_interactions(ratings_file(["1 10 4.0 100"]))
    assert list(df.columns) == ["user_id", "item_id"]


def test_load_interactions_removes_duplicate_pairs(ratings_file):
    path = ratings_file(["1 10 4.0 100", "1 10 2.0 500", "2 20 5.0 200"])
    df = load_interactions(path)
    assert dataframe_to_interactions(df) == [(1, 10), (2, 20)]


def test_load_interactions_drops_rows_with_invalid_ids(ratings_file):
    path = ratings_file(["1 10 4.0 100", "abc 20 5.0 200", "3 xyz 3.0 300"])
    assert dataframe_to_interactions(load_interactions(path)) == [(1, 10)]


def test_load_interactions_keeps_rows_without_timestamp(ratings_file):
    # Offline only needs user and item, so a missing timestamp is fine
    path = ratings_file(["1 10 4.0", "2 20 5.0 200"])
    assert len(load_interactions(path)) == 2


def test_load_interactions_keeps_file_order(ratings_file):
    path = ratings_file(["3 30 3.0 300", "1 10 4.0 100", "2 20 5.0 200"])
    df = load_interactions(path)
    assert dataframe_to_interactions(df) == [(3, 30), (1, 10), (2, 20)]


@pytest.mark.parametrize("limit, expected", [(1, 1), (2, 2), (3, 3), (100, 3)])
def test_load_interactions_respects_limit(ratings_file, limit, expected):
    path = ratings_file(["1 10 4.0 100", "2 20 5.0 200", "3 30 3.0 300"])
    assert len(load_interactions(path, limit_rows=limit)) == expected


def test_limit_is_applied_after_deduplication(ratings_file):
    path = ratings_file(["1 10 4.0 100", "1 10 4.0 150", "2 20 5.0 200"])
    df = load_interactions(path, limit_rows=2)
    assert dataframe_to_interactions(df) == [(1, 10), (2, 20)]


def test_load_interactions_returns_integers(ratings_file):
    df = load_interactions(ratings_file(["1 10 4.0 100"]))
    assert df["user_id"].dtype == "int64"
    assert df["item_id"].dtype == "int64"


# ---------- Online loader: load_events ----------

def test_load_events_sorted_oldest_first(ratings_file):
    path = ratings_file(["1 10 4.0 300", "2 20 5.0 100", "3 30 3.0 200"])
    assert load_events(path)["timestamp"].tolist() == [100, 200, 300]


def test_load_events_keeps_duplicate_pairs(ratings_file):
    path = ratings_file(["1 10 4.0 100", "1 10 2.0 500"])
    assert len(load_events(path)) == 2


def test_load_events_limit_keeps_the_oldest(ratings_file):
    path = ratings_file(["1 10 4.0 300", "2 20 5.0 100", "3 30 3.0 200"])
    df = load_events(path, limit_rows=2)
    assert df["timestamp"].tolist() == [100, 200]


def test_load_events_drops_rows_without_timestamp(ratings_file):
    path = ratings_file(["1 10 4.0", "2 20 5.0 200"])
    assert dataframe_to_events(load_events(path)) == [(2, 20, 200)]


def test_load_events_equal_timestamps_keep_file_order(ratings_file):
    path = ratings_file(["1 10 4.0 100", "2 20 5.0 100", "3 30 3.0 100"])
    assert load_events(path)["user_id"].tolist() == [1, 2, 3]


# ---------- Shared behaviour ----------

@pytest.mark.parametrize("loader", [load_interactions, load_events])
@pytest.mark.parametrize("bad_limit", [0, -1])
def test_invalid_limit_raises(ratings_file, loader, bad_limit):
    path = ratings_file(["1 10 4.0 100"])
    with pytest.raises(ValueError):
        loader(path, limit_rows=bad_limit)


# ---------- Conversion helpers ----------

def test_dataframe_to_interactions_gives_plain_tuples(ratings_file):
    result = dataframe_to_interactions(load_interactions(ratings_file(["1 10 4.0 100"])))
    assert result == [(1, 10)]
    assert isinstance(result[0], tuple)


def test_get_unique_users_items_sorted_and_unique():
    users, items = get_unique_users_items([(2, 20), (1, 10), (2, 10)])
    assert users == [1, 2]
    assert items == [10, 20]
