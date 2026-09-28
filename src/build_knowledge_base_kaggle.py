

import os
import pandas as pd
import chromadb

CLEAN_PATH = "data/products_clean.csv"
DB_PATH = "chroma_db"
COLLECTION_NAME = "iris_kaggle_products"

client = chromadb.PersistentClient(path=DB_PATH)

# get_or_create so re-running the script doesn't crash on "collection already exists" - it just reuses it.
collection = client.get_or_create_collection(name=COLLECTION_NAME)


existing_count = collection.count()

if existing_count > 0:
    print(f"Collection already has {existing_count} products - skipping reload.")
    print("(Delete the 'chroma_db' folder if you want to rebuild from scratch.)")
else:
    products = pd.read_csv(CLEAN_PATH)
    print(f"Loaded {len(products)} cleaned products from {CLEAN_PATH}")

    BATCH_SIZE = 500
    documents = products["description"].tolist()
    ids = [f"product_{i}" for i in range(len(products))]
    metadatas = [
        {
            "source": "kaggle",
            "brand": str(row["brand"]),
            "product_id": str(row["id"]),
            "price_min": float(row["price_min"]),
            "price_max": float(row["price_max"]),
        }
        for _, row in products.iterrows()
    ]

    for i in range(0, len(documents), BATCH_SIZE):
        collection.add(
            documents=documents[i:i + BATCH_SIZE],
            ids=ids[i:i + BATCH_SIZE],
            metadatas=metadatas[i:i + BATCH_SIZE],
        )
        print(f"Loaded batch {i} to {min(i + BATCH_SIZE, len(documents))}")

    print(f"\nDone. Total products in knowledge base: {len(documents)}")

if __name__ == "__main__":
    test_question = "bluetooth speaker under 70 dollars"
    print(f"\nTest query: '{test_question}'")

    results = collection.query(query_texts=[test_question], n_results=5)
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        print(f"  (score {dist:.3f}) {doc}")