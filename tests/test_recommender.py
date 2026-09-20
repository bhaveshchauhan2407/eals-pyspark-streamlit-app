"""Tests for src/recommender.py: turning model scores into a top-K list."""

import numpy as np
import pytest

from src.config import FastALSConfig
from src.model import FastALSModel
from src.recommender import recommend_top_k

# User 100 has already seen items 10 and 20.
INTERACTIONS = [(100, 10), (100, 20), (200, 30), (200, 40), (300, 50)]


@pytest.fixture
def model():
    """A tiny model whose scores we set by hand, so we know the right answer.

    With a single latent factor and every user vector equal to [1], each
    item's score is simply its own value in V:
        item 10 -> 1, item 20 -> 2, item 30 -> 5, item 40 -> 3, item 50 -> 4
    """
    config = FastALSConfig(factors=1, show_progress=False, show_loss=False)
    m = FastALSModel(interactions=INTERACTIONS, config=config)
    m.U = np.ones((m.user_count, 1))
    m.V = np.array([[1.0], [2.0], [5.0], [3.0], [4.0]])  # items 10, 20, 30, 40, 50
    return m


def test_recommendations_sorted_by_score(model):
    # Unseen items for user 100: 30 (score 5), 50 (score 4), 40 (score 3)
    assert recommend_top_k(model, 100, k=3) == [30, 50, 40]


def test_already_seen_items_are_never_recommended(model):
    recs = recommend_top_k(model, 100, k=10)
    assert 10 not in recs
    assert 20 not in recs


@pytest.mark.parametrize("k, expected_length", [(1, 1), (2, 2), (3, 3), (10, 3)])
def test_returns_at_most_k_items(model, k, expected_length):
    # User 100 has only 3 unseen items, so asking for 10 still gives 3
    assert len(recommend_top_k(model, 100, k=k)) == expected_length


def test_unknown_user_gets_no_recommendations(model):
    assert recommend_top_k(model, 999, k=5) == []


def test_returns_original_item_ids_not_internal_indices(model):
    # Internally items are numbered 0-4; the output must use the real ids
    recs = recommend_top_k(model, 100, k=3)
    assert set(recs) <= {10, 20, 30, 40, 50}
