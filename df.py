"""
df.py - A lightweight DataFrame implementation for tabular data operations.
"""


class DataFrame:
    """A simple DataFrame for storing and manipulating tabular data."""

    def __init__(self, data=None):
        """
        Initialize a DataFrame.

        Args:
            data: A dict mapping column names to lists of values,
                  or None to create an empty DataFrame.
        """
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError("data must be a dict mapping column names to lists")
        lengths = [len(v) for v in data.values()]
        has_inconsistent_lengths = lengths and len(set(lengths)) != 1
        if has_inconsistent_lengths:
            raise ValueError("All columns must have the same length")
        self._data = {k: list(v) for k, v in data.items()}

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def columns(self):
        """Return the list of column names."""
        return list(self._data.keys())

    @property
    def shape(self):
        """Return (num_rows, num_cols) as a tuple."""
        num_cols = len(self._data)
        num_rows = len(next(iter(self._data.values()))) if self._data else 0
        return (num_rows, num_cols)

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        if not self._data:
            return "DataFrame(empty)"
        rows, cols = self.shape
        header = " | ".join(f"{c:>10}" for c in self.columns)
        sep = "-+-".join("-" * 10 for _ in self.columns)
        lines = [header, sep]
        for i in range(rows):
            row = " | ".join(f"{self._data[c][i]:>10}" for c in self.columns)
            lines.append(row)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Column access
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        """
        Get a column by name (returns a list) or a list of columns
        (returns a new DataFrame).
        """
        if isinstance(key, str):
            if key not in self._data:
                raise KeyError(f"Column '{key}' not found")
            return list(self._data[key])
        if isinstance(key, list):
            for k in key:
                if k not in self._data:
                    raise KeyError(f"Column '{k}' not found")
            return DataFrame({k: self._data[k] for k in key})
        raise TypeError(f"key must be str or list, not {type(key).__name__}")

    def __setitem__(self, key, values):
        """Add or replace a column."""
        values = list(values)
        if self._data:
            expected = self.shape[0]
            if len(values) != expected:
                raise ValueError(
                    f"Column length {len(values)} does not match DataFrame length {expected}"
                )
        self._data[key] = values

    # ------------------------------------------------------------------
    # Row operations
    # ------------------------------------------------------------------

    def head(self, n=5):
        """Return the first *n* rows as a new DataFrame."""
        return DataFrame({k: v[:n] for k, v in self._data.items()})

    def tail(self, n=5):
        """Return the last *n* rows as a new DataFrame."""
        return DataFrame({k: v[-n:] for k, v in self._data.items()})

    def filter(self, predicate):
        """
        Return rows where *predicate* is True.

        Args:
            predicate: A callable that receives a row dict and returns bool,
                       or a boolean list/mask of the same length as the DataFrame.

        Returns:
            A new DataFrame with the matching rows.
        """
        if callable(predicate):
            rows, _ = self.shape
            mask = [predicate(self._row(i)) for i in range(rows)]
        else:
            mask = list(predicate)

        def _keep_rows(values):
            return [v for v, m in zip(values, mask) if m]

        return DataFrame({k: _keep_rows(vals) for k, vals in self._data.items()})

    def _row(self, index):
        """Return the row at *index* as a dict."""
        return {k: v[index] for k, v in self._data.items()}

    def iterrows(self):
        """Yield (index, row_dict) pairs."""
        for i in range(len(self)):
            yield i, self._row(i)

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def _numeric_column(self, col):
        values = self._data[col]
        try:
            return [float(v) for v in values]
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Column '{col}' contains non-numeric values") from exc

    def sum(self, col):
        """Return the sum of a numeric column."""
        return sum(self._numeric_column(col))

    def mean(self, col):
        """Return the mean of a numeric column."""
        nums = self._numeric_column(col)
        if not nums:
            return None
        return sum(nums) / len(nums)

    def min(self, col):
        """Return the minimum value of a column."""
        return min(self._data[col])

    def max(self, col):
        """Return the maximum value of a column."""
        return max(self._data[col])

    def count(self, col=None):
        """Return the number of rows (optionally for a named column)."""
        if col is None:
            return len(self)
        if col not in self._data:
            raise KeyError(f"Column '{col}' not found")
        return sum(1 for v in self._data[col] if v is not None)

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    def sort(self, col, ascending=True):
        """
        Return a new DataFrame sorted by *col*.

        Args:
            col: Column name to sort by.
            ascending: Sort order (default True).

        Returns:
            A new sorted DataFrame.
        """
        if col not in self._data:
            raise KeyError(f"Column '{col}' not found")
        indices = sorted(range(len(self)), key=lambda i: self._data[col][i],
                         reverse=not ascending)
        return DataFrame({k: [v[i] for i in indices] for k, v in self._data.items()})

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_records(cls, records):
        """
        Create a DataFrame from a list of dicts (records).

        Args:
            records: List of dicts, each representing one row.

        Returns:
            A new DataFrame.
        """
        if not records:
            return cls()
        keys = list(records[0].keys())
        data = {k: [r[k] for r in records] for k in keys}
        return cls(data)

    def to_records(self):
        """Return the DataFrame as a list of row dicts."""
        return [self._row(i) for i in range(len(self))]

    @classmethod
    def from_csv(cls, path, delimiter=","):
        """
        Read a CSV file and return a DataFrame.

        Args:
            path: Path to the CSV file.
            delimiter: Field delimiter (default ',').

        Returns:
            A new DataFrame.
        """
        import csv

        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            records = list(reader)
        return cls.from_records(records)

    def to_csv(self, path, delimiter=","):
        """
        Write the DataFrame to a CSV file.

        Args:
            path: Destination file path.
            delimiter: Field delimiter (default ',').
        """
        import csv

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.columns, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(self.to_records())
