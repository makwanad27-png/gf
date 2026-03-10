"""Tests for df.py DataFrame implementation."""

import os
import tempfile
import pytest

from df import DataFrame


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_empty_dataframe():
    df = DataFrame()
    assert df.columns == []
    assert df.shape == (0, 0)
    assert len(df) == 0


def test_construction_from_dict():
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.columns == ["a", "b"]
    assert df.shape == (3, 2)
    assert len(df) == 3


def test_construction_type_error():
    with pytest.raises(TypeError):
        DataFrame([1, 2, 3])


def test_construction_length_mismatch():
    with pytest.raises(ValueError):
        DataFrame({"a": [1, 2], "b": [1, 2, 3]})


# ---------------------------------------------------------------------------
# Column access
# ---------------------------------------------------------------------------

def test_getitem_single_column():
    df = DataFrame({"x": [10, 20, 30]})
    assert df["x"] == [10, 20, 30]


def test_getitem_multiple_columns():
    df = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    sub = df[["a", "c"]]
    assert isinstance(sub, DataFrame)
    assert sub.columns == ["a", "c"]
    assert sub["a"] == [1, 2]


def test_getitem_missing_column():
    df = DataFrame({"a": [1]})
    with pytest.raises(KeyError):
        _ = df["missing"]


def test_setitem_new_column():
    df = DataFrame({"a": [1, 2, 3]})
    df["b"] = [4, 5, 6]
    assert df["b"] == [4, 5, 6]
    assert df.shape == (3, 2)


def test_setitem_wrong_length():
    df = DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        df["b"] = [1, 2]


# ---------------------------------------------------------------------------
# Row operations
# ---------------------------------------------------------------------------

def test_head():
    df = DataFrame({"n": list(range(10))})
    h = df.head(3)
    assert h["n"] == [0, 1, 2]


def test_tail():
    df = DataFrame({"n": list(range(10))})
    t = df.tail(3)
    assert t["n"] == [7, 8, 9]


def test_filter_callable():
    df = DataFrame({"v": [1, 2, 3, 4, 5]})
    result = df.filter(lambda row: row["v"] > 3)
    assert result["v"] == [4, 5]


def test_filter_mask():
    df = DataFrame({"v": [10, 20, 30]})
    result = df.filter([True, False, True])
    assert result["v"] == [10, 30]


def test_iterrows():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    rows = list(df.iterrows())
    assert rows[0] == (0, {"a": 1, "b": 3})
    assert rows[1] == (1, {"a": 2, "b": 4})


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def test_sum():
    df = DataFrame({"v": [1, 2, 3, 4]})
    assert df.sum("v") == 10


def test_mean():
    df = DataFrame({"v": [1, 2, 3, 4]})
    assert df.mean("v") == 2.5


def test_mean_empty():
    df = DataFrame({"v": []})
    assert df.mean("v") is None


def test_min_max():
    df = DataFrame({"v": [3, 1, 4, 1, 5]})
    assert df.min("v") == 1
    assert df.max("v") == 5


def test_count():
    df = DataFrame({"v": [1, None, 3]})
    assert df.count("v") == 2
    assert df.count() == 3


def test_numeric_column_type_error():
    df = DataFrame({"v": ["a", "b"]})
    with pytest.raises(TypeError):
        df.sum("v")


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def test_sort_ascending():
    df = DataFrame({"v": [3, 1, 2]})
    s = df.sort("v")
    assert s["v"] == [1, 2, 3]


def test_sort_descending():
    df = DataFrame({"v": [3, 1, 2]})
    s = df.sort("v", ascending=False)
    assert s["v"] == [3, 2, 1]


def test_sort_missing_column():
    df = DataFrame({"v": [1, 2]})
    with pytest.raises(KeyError):
        df.sort("missing")


# ---------------------------------------------------------------------------
# Records / I/O
# ---------------------------------------------------------------------------

def test_from_records():
    records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    df = DataFrame.from_records(records)
    assert df["a"] == [1, 3]
    assert df["b"] == [2, 4]


def test_from_records_empty():
    df = DataFrame.from_records([])
    assert df.columns == []


def test_to_records():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    records = df.to_records()
    assert records == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]


def test_csv_roundtrip():
    df = DataFrame({"name": ["Alice", "Bob"], "score": ["90", "85"]})
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as fh:
        path = fh.name
    try:
        df.to_csv(path)
        df2 = DataFrame.from_csv(path)
        assert df2["name"] == ["Alice", "Bob"]
        assert df2["score"] == ["90", "85"]
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr_empty():
    df = DataFrame()
    assert "empty" in repr(df)


def test_repr_nonempty():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    r = repr(df)
    assert "a" in r
    assert "b" in r
