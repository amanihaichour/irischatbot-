"""
load_faqs.py

Adds FAQ content into the SAME Chroma collection as the product data
(iris_kaggle_products), so a single search can surface either a
product or a relevant FAQ depending on what the customer asks.
Each FAQ is tagged source="faq" so you can always tell which is which,
same pattern as the earlier fake/real product data tagging.

Run with:
    python src/load_faqs.py
"""

import json
import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="iris_kaggle_products")

with open("data/FAQs.json", "r", encoding="utf-8") as f:
    faqs = json.load(f)

documents = [f"Q: {faq['question']} R: {faq['answer']}" for faq in faqs]
ids = [f"faq_{i}" for i in range(len(faqs))]
metadatas = [{"source": "faq", "category": "faq"} for _ in faqs]

collection.add(documents=documents, ids=ids, metadatas=metadatas)
print(f"Loaded {len(faqs)} FAQ entries into the knowledge base.")