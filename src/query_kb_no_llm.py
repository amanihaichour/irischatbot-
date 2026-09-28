"""
query_kb_no_llm.py

Same goal as query_kb.py (handle French + respect price constraints)
but WITHOUT any LLM call - fully free, no API key, works offline
for the regex part.

  1. FRENCH -> ENGLISH
     Uses deep-translator (wraps Google Translate's free web endpoint,
     no API key needed). Needs internet, but no cost, no quota to
     manage like an LLM API.

  2. CONSTRAINTS (price, brand)
     Plain regex patterns. Less flexible than an LLM (only catches
     patterns you've explicitly written), but 100% predictable and
     free - no risk of the LLM misreading a number.

Run with:
    pip install deep-translator chromadb
    python src/query_kb_no_llm.py
"""

import re
import chromadb
from deep_translator import MyMemoryTranslator

client_db = chromadb.PersistentClient(path="chroma_db")
collection = client_db.get_collection(name="iris_kaggle_products")


def translate_to_english(text: str) -> str:
    try:
        result = MyMemoryTranslator(source="french", target="english").translate(text)
        return result
    except Exception as e:
        print(f"  (translation failed, using original text: {e})")
        return text


# Patterns for "under/below/less than X dollars" in English AND French
PRICE_MAX_PATTERNS = [
    r"under\s*\$?(\d+)",
    r"below\s*\$?(\d+)",
    r"less than\s*\$?(\d+)",
    r"moins de\s*\$?(\d+)",
    r"a moins de\s*\$?(\d+)",
    r"pas plus de\s*\$?(\d+)",
]

PRICE_MIN_PATTERNS = [
    r"over\s*\$?(\d+)",
    r"above\s*\$?(\d+)",
    r"more than\s*\$?(\d+)",
    r"plus de\s*\$?(\d+)",
    r"au moins\s*\$?(\d+)",
]


def extract_price_constraints(text: str):
    text_lower = text.lower()
    price_max = None
    price_min = None

    for pattern in PRICE_MAX_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            price_max = float(match.group(1))
            text_lower = text_lower[:match.start()] + text_lower[match.end():]
            break

    for pattern in PRICE_MIN_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            price_min = float(match.group(1))
            text_lower = text_lower[:match.start()] + text_lower[match.end():]
            break

    return price_min, price_max, text_lower.strip()


def extract_brand(text: str, known_brands: list[str]):
    text_lower = text.lower()
    for brand in known_brands:
        if brand.lower() in text_lower:
            return brand
    return None


# Pull the actual list of brands present in the database once, at startup
def get_known_brands() -> list[str]:
    all_data = collection.get(limit=5000)  # metadata only, cheap
    brands = {m["brand"] for m in all_data["metadatas"] if m.get("brand") and m["brand"] != "nan"}
    return sorted(brands)


KNOWN_BRANDS = get_known_brands()
print(f"Loaded {len(KNOWN_BRANDS)} known brands for matching.")


def build_where_filter(price_min, price_max, brand):
    conditions = []
    if price_max is not None:
        conditions.append({"price_max": {"$lte": price_max}})
    if price_min is not None:
        conditions.append({"price_min": {"$gte": price_min}})
    if brand:
        conditions.append({"brand": {"$eq": brand}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def search(raw_question: str, n_results: int = 5):
    print(f"\nQuestion: {raw_question}")

    translated = translate_to_english(raw_question)
    print(f"Translated: {translated}")

    price_min, price_max, remaining_text = extract_price_constraints(translated)
    brand = extract_brand(remaining_text, KNOWN_BRANDS)
    if brand:
        remaining_text = remaining_text.lower().replace(brand.lower(), "").strip()

    print(f"Extracted -> price_min={price_min}, price_max={price_max}, brand={brand}")
    print(f"Search text: {remaining_text!r}")

    where_filter = build_where_filter(price_min, price_max, brand)
    print(f"Where filter: {where_filter}")

    results = collection.query(
        query_texts=[remaining_text],
        n_results=n_results,
        where=where_filter,
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if not docs:
        print("  No results matched the constraints.")
    for doc, meta, dist in zip(docs, metas, dists):
        print(f"  (score {dist:.3f}) [{meta['price_min']}-{meta['price_max']} USD] {doc}")


if __name__ == "__main__":
    search("prix ecouteurs")
    search("ecouteurs prix")
    search("enceinte bluetooth a moins de 70 dollars")