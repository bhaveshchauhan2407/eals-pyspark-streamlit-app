"""Shared test helpers, automatically available to every test file."""

import pytest


@pytest.fixture
def ratings_file(tmp_path):
    """Return a function that writes lines into a temporary ratings file.

    Each test gets its own empty temporary folder (tmp_path), which pytest
    deletes afterwards, so tests never interfere with each other.
    """

    def _write(lines):
        path = tmp_path / "ratings.txt"
        path.write_text("\n".join(lines) + "\n")
        return str(path)

    return _write
