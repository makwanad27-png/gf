# gf

A lightweight Python DataFrame (`df`) for working with tabular data — no
external dependencies required.

## Features

- Create DataFrames from dicts, lists of records, or CSV files
- Column access and assignment
- Row filtering (callable predicate or boolean mask)
- Aggregations: `sum`, `mean`, `min`, `max`, `count`
- Sorting by column (ascending or descending)
- `head` / `tail` row slicing
- `iterrows` iteration
- CSV import / export

## Quick start

```python
from df import DataFrame

# Create from a dict
df = DataFrame({
    "name":  ["Alice", "Bob", "Carol"],
    "score": [92, 85, 78],
})

# Filter rows
high = df.filter(lambda row: row["score"] >= 85)
print(high["name"])   # ['Alice', 'Bob']

# Aggregations
print(df.mean("score"))   # 85.0
print(df.max("score"))    # 92

# Sort
print(df.sort("score", ascending=False)["name"])  # ['Alice', 'Bob', 'Carol']

# CSV round-trip
df.to_csv("scores.csv")
df2 = DataFrame.from_csv("scores.csv")
```

## Running tests

```bash
python -m pytest test_df.py -v
```