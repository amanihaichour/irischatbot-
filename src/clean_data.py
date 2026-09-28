"""
clean_data.py

Run this ONCE (or whenever the raw dataset changes) to produce a
clean CSV. Every other script (loading into Chroma, testing, etc.)
reads from data/products_clean.csv instead of redoing this work.

Run with:
    python src/clean_data.py
"""

import pandas as pd

RAW_PATH = "data/electronics.csv"
CLEAN_PATH = "data/products_clean.csv"

df = pd.read_csv(RAW_PATH)

keep_columns = [
    "id", "name", "brand", "categories", "primaryCategories",
    "prices.amountMin", "prices.amountMax", "prices.currency",
]
df = df[keep_columns]

print(f"Loaded {len(df)} raw rows (before deduplication).")

def aggregate_product(group: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "name": group["name"].iloc[0],
        "brand": group["brand"].iloc[0],
        "categories": group["categories"].iloc[0],
        "primaryCategories": group["primaryCategories"].iloc[0],
        "price_min": group["prices.amountMin"].min(),
        "price_max": group["prices.amountMax"].max(),
        "currency": group["prices.currency"].iloc[0],
        "num_observations": len(group),
    })


products = df.groupby("id").apply(aggregate_product, include_groups=False).reset_index()
print(f"Reduced to {len(products)} unique products after deduplication.")

def build_description(row) -> str:
    name = row["name"]
    brand = row["brand"] if pd.notna(row["brand"]) else "Unknown brand"

    if pd.notna(row["categories"]):
        cats = row["categories"].split(",")[:3]
        category_text = ", ".join(c.strip() for c in cats)
    else:
        category_text = row["primaryCategories"] if pd.notna(row["primaryCategories"]) else "Electronics"

    if row["price_min"] == row["price_max"]:
        price_text = f"{row['price_min']} {row['currency']}"
    else:
        price_text = f"{row['price_min']} to {row['price_max']} {row['currency']}"

    return f"{name} (brand: {brand}, category: {category_text}). Price: {price_text}."


products["description"] = products.apply(build_description, axis=1)

products.to_csv(CLEAN_PATH, index=False)
print(f"\nSaved cleaned data to {CLEAN_PATH}")
print(f"Columns: {list(products.columns)}")
print("\nSample descriptions:")
for desc in products["description"].head(3):
    print(f"  - {desc}")