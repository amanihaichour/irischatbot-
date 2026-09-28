"""
explore_dataset.py

Before loading anything into Chroma, let's actually look at what
we have: columns, missing values, a few sample rows. This is a
good habit for any real dataset, not just this one - assuming
structure without checking is how bugs sneak in later.

Run with:
    pip install pandas
    python src/explore_dataset.py
"""

import pandas as pd

df = pd.read_csv("data/electronics.csv")

print("Shape (rows, columns):", df.shape)
print("\nColumn names:")
print(list(df.columns))

print("\nFirst 3 rows:")
print(df.head(3).to_string())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nSample of unique categories (if a 'categories' column exists):")
if "categories" in df.columns:
    print(df["categories"].dropna().unique()[:10])