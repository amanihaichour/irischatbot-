"""
app.py

A real chat interface for your RAG chatbot, built with Streamlit.
Reuses the exact same pipeline logic from chatbot.py (translation,
constraint extraction, Chroma search, Gemini answer generation) -
this file only adds the visual chat interface on top.

Run with:
    pip install streamlit
    export GEMINI_API_KEY="your-key-here"
    streamlit run src/app.py

This opens a browser tab with a real chat window - type a question,
get an answer, keeps conversation history on screen. This is what
you'd actually show your manager instead of a terminal.
"""

import re
import streamlit as st
import chromadb
from deep_translator import MyMemoryTranslator
from google import genai

st.set_page_config(page_title="IRIS Assistant", page_icon="🛍️", layout="centered")

@st.cache_resource
def get_clients():
    llm = genai.Client()
    db = chromadb.PersistentClient(path="chroma_db")
    collection = db.get_collection(name="iris_kaggle_products")
    return llm, collection


@st.cache_resource
def get_known_brands(_collection):
    all_data = _collection.get(limit=5000)
    brands = {m["brand"] for m in all_data["metadatas"] if m.get("brand") and m["brand"] != "nan"}
    return sorted(brands)


client_llm, collection = get_clients()
KNOWN_BRANDS = get_known_brands(collection)
MODEL = "gemini-3.6-flash"

DOMAIN_GLOSSARY = {
    "support tv": "tv wall mount",
    "support mural": "wall mount",
    "support ecran": "screen mount",
}

CONFIDENCE_THRESHOLD = 1.0

PRICE_MAX_PATTERNS = [r"under\s*\$?(\d+)", r"below\s*\$?(\d+)", r"less than\s*\$?(\d+)"]
PRICE_MIN_PATTERNS = [r"over\s*\$?(\d+)", r"above\s*\$?(\d+)", r"more than\s*\$?(\d+)"]


def apply_domain_glossary(text: str) -> str:
    result = text.lower()
    for phrase, replacement in DOMAIN_GLOSSARY.items():
        result = result.replace(phrase, replacement)
    return result


def translate_to_english(text: str) -> str:
    try:
        result = MyMemoryTranslator(source="french", target="english").translate(text)
        if not result or not result.strip():
            return text
        return result
    except Exception:
        return text


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
    results = collection.query(query_texts=[remaining_text], n_results=n_results, where=where_filter)
    return results["documents"][0], results["distances"][0], where_filter


def generate_answer(original_question: str, matched_products: list[str]) -> str:
    context = "\n".join(f"- {p}" for p in matched_products) if matched_products else "Aucun produit ne correspond."
    prompt = f"""Tu es l'assistant client d'une boutique electronique. Reponds UNIQUEMENT
en te basant sur les produits ci-dessous. N'invente aucun produit ni prix. Si aucun
produit ne correspond, dis-le clairement. Reponds de maniere naturelle et concise,
dans la meme langue que la question du client.

Produits disponibles:
{context}

Question du client: {original_question}"""
    try:
        response = client_llm.models.generate_content(model=MODEL, contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"Desole, le service est temporairement indisponible ({e})."


def generate_clarification(original_question: str) -> str:
    prompt = f"""Tu es l'assistant client d'une boutique electronique. La question du
client est ambigue. Demande UNE precision courte et naturelle, dans la meme langue
que sa question. Ne propose pas de produit, pose juste la question.

Question du client: {original_question}"""
    try:
        response = client_llm.models.generate_content(model=MODEL, contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"Desole, le service est temporairement indisponible ({e})."


def get_bot_response(raw_question: str) -> str:
    matched_products, distances, where_filter = search_products(raw_question)
    best_score = distances[0] if distances else float("inf")

    if where_filter is None and best_score > CONFIDENCE_THRESHOLD:
        return generate_clarification(raw_question)
    return generate_answer(raw_question, matched_products)


st.title("🛍️ IRIS Assistant")
st.caption("Assistant intelligent pour vos questions sur nos produits - francais et anglais")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Comment puis-je vous aider aujourd'hui ?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            answer = get_bot_response(user_input)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})