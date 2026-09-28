"""
chatbot.py

The full pipeline, all pieces combined:

  Customer question (French or English, maybe with a price limit/brand)
        |
        v
  1. Translate to English (MyMemory, free)
        |
        v
  2. Extract price/brand constraints (regex, free)
        |
        v
  3. Chroma search:
        - "where" filter for exact constraints (price, brand)
        - semantic search on the remaining text
        |
        v
  4. Gemini generates a natural-language answer from the matched
     products (this is the ONLY paid-tier-adjacent step, and it's
     on Gemini's free tier)
        |
        v
  Final answer, shown to the customer

Run with:
    export GEMINI_API_KEY="your-key-here"
    python src/chatbot.py
"""

import re
import chromadb
from deep_translator import MyMemoryTranslator
from google import genai

client_llm = genai.Client()
MODEL = "gemini-2.5-flash-lite"  # much higher free-tier daily quota than 3.6-flash

client_db = chromadb.PersistentClient(path="chroma_db")
collection = client_db.get_collection(name="iris_kaggle_products")

# Domain-specific disambiguation: some French phrases translate
# literally into something ambiguous or misleading in English.
# "support tv" -> "tv support" reads like tech support, not a mount.
# We fix known cases BEFORE they reach translation/search, same idea
# as the Darija glossary from earlier in the project.
DOMAIN_GLOSSARY = {
    "support tv": "tv wall mount",
    "support mural": "wall mount",
    "support ecran": "screen mount",
}


def apply_domain_glossary(text: str) -> str:
    result = text.lower()
    for phrase, replacement in DOMAIN_GLOSSARY.items():
        result = result.replace(phrase, replacement)
    return result


# If the best semantic match is above this distance, we treat it as
# "not confident enough" and ask for clarification instead of
# guessing - this is the Phase 4 guardrail from the original plan.
CONFIDENCE_THRESHOLD = 1.0


def get_known_brands() -> list[str]:
    all_data = collection.get(limit=5000)
    brands = {m["brand"] for m in all_data["metadatas"] if m.get("brand") and m["brand"] != "nan"}
    return sorted(brands)


KNOWN_BRANDS = get_known_brands()


def translate_to_english(text: str) -> str:
    try:
        result = MyMemoryTranslator(source="french", target="english").translate(text)
        if not result or not result.strip():
            return text
        return result
    except Exception as e:
        return text


PRICE_MAX_PATTERNS = [
    r"under\s*\$?(\d+)", r"below\s*\$?(\d+)", r"less than\s*\$?(\d+)",
]
PRICE_MIN_PATTERNS = [
    r"over\s*\$?(\d+)", r"above\s*\$?(\d+)", r"more than\s*\$?(\d+)",
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


def search_products(raw_question: str, n_results: int = 5):
    pre_processed = apply_domain_glossary(raw_question)
    translated = translate_to_english(pre_processed)

    price_min, price_max, remaining_text = extract_price_constraints(translated)
    brand = extract_brand(remaining_text, KNOWN_BRANDS)
    if brand:
        remaining_text = remaining_text.lower().replace(brand.lower(), "").strip()

    where_filter = build_where_filter(price_min, price_max, brand)

    results = collection.query(
        query_texts=[remaining_text],
        n_results=n_results,
        where=where_filter,
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    return documents, distances, where_filter


def generate_answer(original_question: str, matched_products: list[str]) -> str:
    if not matched_products:
        context = "Aucun produit ne correspond aux criteres demandes."
    else:
        context = "\n".join(f"- {p}" for p in matched_products)

    prompt = f"""Tu es l'assistant client d'une boutique electronique. Reponds UNIQUEMENT
en te basant sur les produits ci-dessous. N'invente aucun produit ni prix qui n'y
figure pas. Si aucun produit ne correspond, dis-le clairement. Reponds de maniere
naturelle et concise, dans la meme langue que la question du client.

Produits disponibles:
{context}

Question du client: {original_question}"""

    try:
        response = client_llm.models.generate_content(model=MODEL, contents=prompt)
        return response.text.strip()
    except Exception:
        return "Desole, le service est temporairement indisponible. Merci de reessayer dans un instant."


def generate_clarification(original_question: str) -> str:
    prompt = f"""Tu es l'assistant client d'une boutique electronique. La question
du client est ambigue ou nous n'avons pas trouve de produit suffisamment pertinent.
Demande UNE precision courte et naturelle pour mieux comprendre ce qu'il cherche,
dans la meme langue que sa question. Ne propose pas de produit, pose juste la
question de clarification.

Question du client: {original_question}"""

    response = client_llm.models.generate_content(model=MODEL, contents=prompt)
    return response.text.strip()


def ask(raw_question: str):
    print(f"\n{'='*60}")
    print(f"Client: {raw_question}")

    matched_products, distances, where_filter = search_products(raw_question)

    best_score = distances[0] if distances else float("inf")

    # If we have a real filter (price/brand), trust the results even
    # if semantic score is mediocre - the filter already did the
    # precise part. Only fall back to clarification for PURE semantic
    # search with a weak top match.
    if where_filter is None and best_score > CONFIDENCE_THRESHOLD:
        answer = generate_clarification(raw_question)
    else:
        answer = generate_answer(raw_question, matched_products)

    print(f"Assistant: {answer}")


if __name__ == "__main__":
    ask("support mural pour tv")
    ask("prix ecouteurs")
    ask("enceinte bluetooth a moins de 70 dollars")
    ask("Sony speakers under 50 dollars")
    ask("je cherche un support tv")