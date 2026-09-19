"""
Data loading and filtering for the Yelp ratings file.

Pandas replacement for the Spark loaders of the original M1 project
(src/spark_loader.py). 
"""

from typing import List, Tuple

import pandas as pd

COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def read_raw_ratings(path: str) -> pd.DataFrame:
    """
    Read the raw file into 4 columns.
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        dtype=str,
        on_bad_lines="skip",
    )
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_interactions(path: str, limit_rows: int = 10000) -> pd.DataFrame:
    """Offline protocol: unique (user_id, item_id) pairs, in file order.

    The rating is discarded (implicit feedback: only the fact that an
    interaction happened matters) and duplicate pairs count once.
    """
    if limit_rows <= 0:
        raise ValueError("limit_rows must be a positive integer")

    df = read_raw_ratings(path)
    df = (
        df[["user_id", "item_id"]]
        .dropna()
        .drop_duplicates()
        .head(limit_rows)
        .astype("int64")
        .reset_index(drop=True)
    )
    return df


def load_events(path: str, limit_rows: int = 10000) -> pd.DataFrame:
    """Online protocol: (user_id, item_id, timestamp) events, oldest first.

    Duplicates are kept on purpose: the same user interacting with the
    same item at different times is a genuine new event.
    """
    if limit_rows <= 0:
        raise ValueError("limit_rows must be a positive integer")

    df = read_raw_ratings(path).dropna()
    df = (
        df.sort_values("timestamp", kind="stable")
        .head(limit_rows)[["user_id", "item_id", "timestamp"]]
        .astype("int64")
        .reset_index(drop=True)
    )
    return df


def dataframe_to_interactions(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Convert to the list of (user, item) tuples the model expects."""
    return list(df[["user_id", "item_id"]].itertuples(index=False, name=None))


def dataframe_to_events(df: pd.DataFrame) -> List[Tuple[int, int, int]]:
    """Convert to the list of (user, item, timestamp) tuples for the online split."""
    return list(
        df[["user_id", "item_id", "timestamp"]].itertuples(index=False, name=None)
    )


def get_unique_users_items(interactions: List[Tuple[int, int]]):
    users = sorted({u for u, _ in interactions})
    items = sorted({i for _, i in interactions})
    return users, items