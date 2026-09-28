"""
api.py - FastAPI backend for the IRIS chatbot.

Two ways in:
  1. POST /chat          - the web widget (frontend/index.html)
  2. GET/POST /webhooks/meta - Facebook/Instagram DMs via Meta's Messenger
     Platform, so people can message the IRIS Facebook/Instagram page
     directly and get the same bot.

Both channels share the exact same pipeline (search_products,
generate_answer, etc.) and the exact same conversation-history logic -
no duplicated logic between the web chat and the Meta channel.
"""

import os
import re
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import httpx
import chromadb
import ollama
from deep_translator import MyMemoryTranslator
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="IRIS Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "llama3.2"

# Ollama's first call after a server (re)start has to load the model into
# memory, which can take long enough to time out a real user's first
# message. Warm it up once here so that doesn't happen mid-conversation.
try:
    print("[server log] Warming up Ollama model (first load can be slow)...")
    ollama.chat(model=MODEL, messages=[{"role": "user", "content": "test"}])
    print("[server log] Ollama warm-up complete.")
except Exception as e:
    print(f"[server log] Ollama warm-up failed (will retry on first real request): {e}")

client_db = chromadb.PersistentClient(path="chroma_db")
collection = client_db.get_collection(name="iris_kaggle_products")


def get_known_brands() -> list[str]:
    all_data = collection.get(limit=5000)
    brands = {m["brand"] for m in all_data["metadatas"] if m.get("brand") and m["brand"] != "nan"}
    return sorted(brands)


KNOWN_BRANDS = get_known_brands()

DOMAIN_GLOSSARY = {
    "support tv": "tv wall mount",
    "support mural": "wall mount",
    "support ecran": "screen mount",
    "refrigirator": "refrigerator",
    "frigerator": "refrigerator",
    "frigo": "refrigerateur",
    "frigos": "refrigerateurs",
    "frigidaire": "refrigerateur",
    "frijider": "refrigerateur",
    "frigider": "refrigerateur",
    "clim": "climatiseur",
    "tele": "television",
}

CONFIDENCE_THRESHOLD = 1.0
NO_MATCH_THRESHOLD = 1.3  # above this, don't force an answer - just be honest nothing matches

PRICE_MAX_PATTERNS = [r"under\s*\$?(\d+)", r"below\s*\$?(\d+)", r"less than\s*\$?(\d+)"]
PRICE_MIN_PATTERNS = [r"over\s*\$?(\d+)", r"above\s*\$?(\d+)", r"more than\s*\$?(\d+)"]

# Message categories - "livraison_commande" was dropped since IRIS doesn't
# offer that service. Everything downstream (classification prompt,
# logging, analytics) reads from this single list, so changing it here
# is the only change needed if the manager gives a different final list.
MESSAGE_CATEGORIES = [
    "remerciement",
    "reclamation",
    "demande_info",
    "demande_prix",
    "autre",
]

MESSAGE_CATEGORY_DESCRIPTIONS = {
    "remerciement": "le client remercie ou exprime sa satisfaction",
    "reclamation": "le client signale un probleme, un defaut, une erreur, ou une insatisfaction",
    "demande_info": "le client demande des informations generales sur un produit ou service (hors prix)",
    "demande_prix": "le client demande specifiquement le prix d'un produit",
    "autre": "tout le reste, salutations, hors-sujet",
}


def classify_message(text: str, history: list[dict] = None) -> str:
    """Classify a customer message into one of the supported categories."""
    categories_str = "\n".join(
        f"- {cat}: {MESSAGE_CATEGORY_DESCRIPTIONS[cat]}" for cat in MESSAGE_CATEGORIES
    )
    history_text = format_history(history or [])
    prompt = f"""Classifie le message du client ci-dessous dans EXACTEMENT une de ces categories:
{categories_str}

Exemples (le ton compte plus que le mot "merci" lui-meme - toute expression de
satisfaction ou de plaisir sur un achat est un remerciement, meme sans dire "merci") :
- "merci beaucoup pour votre aide" -> remerciement
- "j'ai achete des ecouteurs et j'adore, super qualite" -> remerciement
- "je suis tres content de mon achat" -> remerciement
- "franchement le produit est top" -> remerciement
- "le produit est arrive casse, c'est inadmissible" -> reclamation
- "je ne suis pas satisfait, ca ne marche pas" -> reclamation
- "vous avez des televisions 4K?" -> demande_info
- "quelles sont les caracteristiques de ce frigo?" -> demande_info
- "combien coute ce produit?" -> demande_prix
- "c'est a combien le prix?" -> demande_prix
- "bonjour" -> autre
- "comment allez-vous?" -> autre

Reponds UNIQUEMENT avec le nom exact de la categorie, rien d'autre, pas de ponctuation,
pas d'explication.

Historique recent (pour le contexte si besoin):
{history_text}

Message du client: {text}"""
    try:
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        result = response["message"]["content"].strip().lower()
        # Fall back instead of storing an LLM-generated category we do not recognize.
        for category in MESSAGE_CATEGORIES:
            if category in result:
                return category
        return "autre"
    except Exception as e:
        print(f"[server log] Classification failed: {e}")
        return "autre"


def extract_product_mentioned(text: str, history: list[dict] = None) -> str | None:
    """Extract the product or product type mentioned by the customer, if any."""
    history_text = format_history(history or [])
    prompt = f"""Quel produit ou type de produit le client mentionne-t-il dans ce message
ou dans l'historique recent ? Reponds avec le nom du produit en quelques mots seulement
(ex: "televiseur Samsung", "frigo", "ecouteurs sans fil"). Si aucun produit specifique
n'est mentionne, reponds exactement "aucun".

Historique recent:
{history_text}

Message: {text}"""
    try:
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        result = response["message"]["content"].strip()
        if result.lower() in ("aucun", "none", ""):
            return None
        return result
    except Exception as e:
        print(f"[server log] Product extraction failed: {e}")
        return None


DB_LOG_PATH = Path("data/chat_logs.db")
DB_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_log_db():
    conn = sqlite3.connect(DB_LOG_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            channel TEXT NOT NULL,
            category TEXT,
            timestamp TEXT NOT NULL,
            user_message TEXT NOT NULL,
            search_query_used TEXT,
            bot_reply TEXT NOT NULL,
            matched_products TEXT,
            filter_used TEXT,
            best_score REAL,
            was_clarification INTEGER,
            customer_name TEXT,
            customer_phone TEXT,
            product_mentioned TEXT
        )
    """)
    # Migration safety net: CREATE TABLE IF NOT EXISTS is a no-op if the
    # table already existed from before these columns were added (this is
    # exactly the bug that bit the `category` column earlier). Add any
    # missing columns explicitly so old chat_logs.db files keep working
    # without needing `rm data/chat_logs.db`.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations)")}
    for col in ("customer_name", "customer_phone", "product_mentioned"):
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE conversations ADD COLUMN {col} TEXT")
    conn.commit()
    conn.close()


init_log_db()


def log_conversation(session_id, channel, category, user_message, search_query_used, bot_reply,
                      matched_products, where_filter, best_score, was_clarification,
                      customer_name=None, customer_phone=None, product_mentioned=None):
    try:
        conn = sqlite3.connect(DB_LOG_PATH)
        conn.execute(
            """INSERT INTO conversations
               (session_id, channel, category, timestamp, user_message, search_query_used, bot_reply,
                matched_products, filter_used, best_score, was_clarification,
                customer_name, customer_phone, product_mentioned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                channel,
                category,
                datetime.now(timezone.utc).isoformat(),
                user_message,
                search_query_used,
                bot_reply,
                json.dumps(matched_products, ensure_ascii=False),
                json.dumps(where_filter, ensure_ascii=False) if where_filter else None,
                best_score,
                int(was_clarification),
                customer_name,
                customer_phone,
                product_mentioned,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[server log] Failed to write conversation log: {e}")


def apply_domain_glossary(text: str) -> str:
    result = text.lower()
    for phrase, replacement in DOMAIN_GLOSSARY.items():
        result = result.replace(phrase, replacement)
    return result


def translate_to_english(text: str) -> str:
    try:
        result = MyMemoryTranslator(source="french", target="english").translate(text)
        if not result or not result.strip():
            print(f"[server log] Translation returned empty for: {text!r}")
            return text
        print(f"[server log] Translated {text!r} -> {result!r}")
        return result
    except Exception as e:
        print(f"[server log] Translation failed for {text!r}: {e}")
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


def format_history(history: list[dict]) -> str:
    if not history:
        return "(Aucun echange precedent.)"
    lines = []
    for turn in history[-6:]:
        role = "Client" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)


def generate_answer(original_question: str, matched_products: list[str], history: list[dict] = None) -> str:
    context = "\n".join(f"- {p}" for p in matched_products) if matched_products else "Aucun produit ne correspond."
    history_text = format_history(history or [])
    prompt = f"""Tu es l'assistant client d'une boutique electronique. Reponds UNIQUEMENT
en te basant sur les produits ci-dessous. N'invente aucun produit ni prix. Reponds de
maniere naturelle et concise, dans la meme langue que la question du client. Utilise
l'historique de conversation ci-dessous pour comprendre le contexte.

Si les produits ci-dessous ne correspondent pas parfaitement a la demande (par exemple
si le client n'a pas donne assez de details), presente quand meme les meilleures options
disponibles plutot que de ne rien proposer, et propose au client d'affiner sa recherche
si besoin. Ne repete pas de question de clarification si le client a deja exprime de
l'impatience ou a deja repondu "je ne sais pas" a une question precedente - dans ce cas,
montre directement des options.

Historique de la conversation:
{history_text}

Produits disponibles:
{context}

Question actuelle du client: {original_question}"""
    try:
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[server log] LLM call failed: {e}")
        return "Desole, notre assistant est temporairement surcharge. Merci de reessayer dans quelques instants."


def generate_clarification(original_question: str, history: list[dict] = None) -> str:
    history_text = format_history(history or [])
    prompt = f"""Tu es l'assistant client d'une boutique electronique. Tu n'as pas trouve
de produit correspondant avec assez de confiance a la question du client.

Pose UNE question de clarification courte et naturelle pour mieux comprendre ce qu'il
cherche (par exemple: quel type de produit, quelle gamme de prix, quelle marque).

REGLES IMPORTANTES:
- Ne commente JAMAIS l'orthographe, la clarte ou le choix des mots du client.
- Ne repete pas ou ne cite pas les mots exacts du client entre guillemets.
- Ne dis jamais des phrases comme "le mot X n'est pas clair" ou "votre message est ambigu".
- Reste naturel, chaleureux et bref, comme le ferait un vendeur humain.
- Reponds dans la meme langue que le client.

Historique de la conversation:
{history_text}

Question actuelle du client: {original_question}"""
    try:
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[server log] LLM call failed: {e}")
        return "Desole, notre assistant est temporairement surcharge. Merci de reessayer dans quelques instants."


def build_search_query(raw_question: str, history: list[dict]) -> str:
    # Short/vague follow-ups ("oui", "non", "et le prix?") carry little
    # useful meaning alone for semantic search. Pull in recent USER
    # turns (not just the single last one) so the original topic isn't
    # lost after several back-and-forth clarification exchanges.
    #
    # NOTE: history sent by the frontend already includes the current
    # message as its last entry (it's pushed before the request is
    # sent), so we exclude that last entry here to avoid duplicating
    # raw_question with itself.
    cleaned = re.sub(r"[?!.,]", "", raw_question.strip())
    if len(cleaned.split()) <= 6 and history:
        previous_turns = history[:-1] if history and history[-1].get("content") == raw_question else history
        recent_user_messages = [
            turn["content"] for turn in previous_turns[-8:] if turn.get("role") == "user"
        ]
        if recent_user_messages:
            return " ".join(recent_user_messages[-3:]) + " " + raw_question
    return raw_question


def generate_no_match_reply(original_question: str, history: list[dict] = None) -> str:
    """Honestly explain when no relevant products are available."""
    history_text = format_history(history or [])
    prompt = f"""Tu es l'assistant client d'une boutique electronique. Apres verification,
aucun produit pertinent ne correspond a la demande du client - le magasin ne semble
pas proposer ce type de produit.

Dis-le clairement et honnetement, sans proposer de produits non lies. Tu peux
mentionner brievement les grandes categories que le magasin propose (electronique,
TV, audio, informatique, etc.) au cas ou cela aiderait le client a reformuler.
Reste bref, naturel et dans la meme langue que le client.

Historique de la conversation:
{history_text}

Question actuelle du client: {original_question}"""
    try:
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[server log] LLM call failed: {e}")
        return "Desole, notre assistant est temporairement surcharge. Merci de reessayer dans quelques instants."


def already_asked_for_clarification(history: list[dict]) -> bool:
    """Check whether the assistant recently asked a clarifying question."""
    for turn in history[-4:]:
        if turn.get("role") == "assistant" and "?" in turn.get("content", ""):
            return True
    return False


def already_asked_for_complaint_details(history: list[dict]) -> bool:
    """Check whether the assistant recently requested complaint contact details."""
    for turn in history[-2:]:
        content = turn.get("content", "").lower()
        if turn.get("role") == "assistant" and (
            "numero" in content or "numéro" in content
            or "telephone" in content or "téléphone" in content
        ):
            return True
    return False


def already_asked_for_thanks_product(history: list[dict]) -> bool:
    """Check whether the assistant recently asked which product was meant."""
    for turn in history[-2:]:
        content = turn.get("content", "").lower()
        if turn.get("role") == "assistant" and (
            "quel produit" in content or "quel article" in content
            or "de quel produit" in content
        ):
            return True
    return False


PHONE_REGEX = r'(?:\+?\d[\d\s\-\.]{7,14}\d)'


def extract_contact_info(text: str):
    """Extract a phone number with regex and a name with the LLM."""
    phone_match = re.search(PHONE_REGEX, text)
    phone = phone_match.group(0).strip() if phone_match else None

    name = None
    try:
        prompt = f"""Extrait UNIQUEMENT le nom de la personne dans ce message, s'il y en a un.
Reponds avec le nom seul, ou "aucun" s'il n'y en a pas.

Message: {text}"""
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        result = response["message"]["content"].strip()
        if result.lower() not in ("aucun", "none", ""):
            name = result
    except Exception as e:
        print(f"[server log] Name extraction failed: {e}")

    return name, phone


def generate_complaint_apology(original_question: str, history: list[dict] = None) -> str:
    history_text = format_history(history or [])
    prompt = f"""Tu es l'assistant client d'une boutique electronique. Le client vient de
signaler un probleme ou une reclamation.

Presente d'abord des excuses sinceres et breves. Ensuite, demande-lui de preciser ce qui
s'est passe et quel produit est concerne (si ce n'est pas deja clair), puis demande-lui
poliment son nom et son numero de telephone afin qu'un membre de l'equipe puisse le
recontacter rapidement.

Reste naturel, chaleureux et bref. Reponds dans la meme langue que le client.

Historique de la conversation:
{history_text}

Message du client: {original_question}"""
    try:
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[server log] LLM call failed: {e}")
        return "Nous sommes vraiment desoles pour ce probleme. Pourriez-vous nous donner votre nom et votre numero de telephone afin qu'un membre de notre equipe vous recontacte ?"


def generate_complaint_confirmation(name, phone, history: list[dict] = None) -> str:
    history_text = format_history(history or [])
    prompt = f"""Tu es l'assistant client d'une boutique electronique. Le client vient de
donner ses coordonnees suite a une reclamation.

Remercie-le, confirme qu'un membre de l'equipe va le recontacter tres bientot, et reste
bref et rassurant. Reponds dans la meme langue que le client.

Historique de la conversation:
{history_text}"""
    try:
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[server log] LLM call failed: {e}")
        return "Merci, nous avons bien note vos coordonnees. Un membre de notre equipe vous recontactera tres bientot."


def generate_thanks_reply(original_question: str, history: list[dict] = None, ask_for_product: bool = False) -> str:
    history_text = format_history(history or [])
    if ask_for_product:
        instruction = (
            'Comme le client n\'a pas precise de produit, demande-lui brievement pour '
            'quel produit il exprime sa satisfaction, en une seule question courte et '
            'naturelle (ex: "De quel produit s\'agit-il ?").'
        )
    else:
        instruction = 'Ne pose pas de question, ce n\'est pas necessaire.'
    prompt = f"""Tu es l'assistant client d'une boutique electronique. Le client vient
d'exprimer un remerciement ou une satisfaction.

Reponds chaleureusement et brievement pour le remercier a ton tour. {instruction}
Reponds dans la meme langue que le client.

Historique de la conversation:
{history_text}

Message du client: {original_question}"""
    try:
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[server log] LLM call failed: {e}")
        if ask_for_product:
            return "Merci beaucoup pour votre retour ! Pourriez-vous nous dire pour quel produit ?"
        return "Merci beaucoup pour votre retour, ca nous fait tres plaisir !"


def get_bot_response(raw_question: str, history: list[dict] = None):
    """Return a bot reply and the associated conversation logging fields."""
    history = history or []

    if already_asked_for_complaint_details(history):
        name, phone = extract_contact_info(raw_question)
        product = extract_product_mentioned(raw_question, history)
        reply = generate_complaint_confirmation(name, phone, history)
        print(f"[server log] Complaint contact captured: name={name} phone={phone} product={product}")
        return (reply, False, raw_question, [], None, 0.0, "reclamation", name, phone, product)

    if already_asked_for_thanks_product(history):
        product = extract_product_mentioned(raw_question, history)
        reply = generate_thanks_reply(raw_question, history, ask_for_product=False)
        print(f"[server log] Thanks product captured: product={product}")
        return (reply, False, raw_question, [], None, 0.0, "remerciement", None, None, product)

    category = classify_message(raw_question, history)

    if category == "reclamation":
        product = extract_product_mentioned(raw_question, history)
        reply = generate_complaint_apology(raw_question, history)
        return (reply, False, raw_question, [], None, 0.0, category, None, None, product)

    if category == "remerciement":
        product = extract_product_mentioned(raw_question, history)
        reply = generate_thanks_reply(raw_question, history, ask_for_product=(product is None))
        return (reply, False, raw_question, [], None, 0.0, category, None, None, product)

    search_query = build_search_query(raw_question, history)
    matched_products, distances, where_filter = search_products(search_query)
    best_score = distances[0] if distances else float("inf")

    should_clarify = (
        where_filter is None
        and best_score > CONFIDENCE_THRESHOLD
        and not already_asked_for_clarification(history)
    )

    if should_clarify:
        reply = generate_clarification(raw_question, history)
        was_clarification = True
    elif where_filter is None and best_score > NO_MATCH_THRESHOLD:
        # Even after giving the customer a chance to clarify, nothing
        # relevant was found - be honest instead of forcing an answer
        # out of unrelated products.
        reply = generate_no_match_reply(raw_question, history)
        was_clarification = False
    else:
        reply = generate_answer(raw_question, matched_products, history)
        was_clarification = False

    return (reply, was_clarification, search_query, matched_products, where_filter,
            best_score, category, None, None, None)


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    session_id: str = "web_session"


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    (reply, was_clarification, search_query, matched_products, where_filter,
     best_score, category, customer_name, customer_phone, product_mentioned) = get_bot_response(
        request.message, request.history
    )
    log_conversation(
        session_id=request.session_id,
        channel="web",
        category=category,
        user_message=request.message,
        search_query_used=search_query,
        bot_reply=reply,
        matched_products=matched_products,
        where_filter=where_filter,
        best_score=best_score,
        was_clarification=was_clarification,
        customer_name=customer_name,
        customer_phone=customer_phone,
        product_mentioned=product_mentioned,
    )
    return ChatResponse(reply=reply)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "IRIS Chatbot API is running"}


# SECURITY: never hardcode real tokens in source code. Read them from
# environment variables instead:
#
META_VERIFY_TOKEN = os.environ.get("META_VERIFY_TOKEN")
META_PAGE_ACCESS_TOKEN = os.environ.get("META_PAGE_ACCESS_TOKEN")
META_GRAPH_API_VERSION = "v21.0"

# Meta gives no built-in conversation history - we keep a small
# in-memory per-sender history here, same shape as the web chat's
# history, so get_bot_response() works unchanged for both channels.
# Resets on server restart; a real deployment would persist this.
meta_conversation_memory: dict[str, list[dict]] = {}


@app.get("/webhooks/meta")
def verify_meta_webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == META_VERIFY_TOKEN:
        print("[server log] Meta webhook verified successfully.")
        return Response(content=challenge, media_type="text/plain")

    raise HTTPException(status_code=403, detail="Verification token mismatch")


@app.post("/webhooks/meta")
async def handle_meta_message(request: Request):
    payload = await request.json()

    try:
        for entry in payload.get("entry", []):
            for messaging_event in entry.get("messaging", []):
                sender_id = messaging_event.get("sender", {}).get("id")
                message_text = messaging_event.get("message", {}).get("text")

                if not sender_id or not message_text:
                    continue

                print(f"[server log] Meta message from {sender_id}: {message_text}")

                history = meta_conversation_memory.get(sender_id, [])

                (reply, was_clarification, search_query, matched_products, where_filter,
                 best_score, category, customer_name, customer_phone, product_mentioned) = get_bot_response(
                    message_text, history
                )

                history.append({"role": "user", "content": message_text})
                history.append({"role": "assistant", "content": reply})
                meta_conversation_memory[sender_id] = history[-12:]

                log_conversation(
                    session_id=sender_id,
                    channel="meta",
                    category=category,
                    user_message=message_text,
                    search_query_used=search_query,
                    bot_reply=reply,
                    matched_products=matched_products,
                    where_filter=where_filter,
                    best_score=best_score,
                    was_clarification=was_clarification,
                    customer_name=customer_name,
                    customer_phone=customer_phone,
                    product_mentioned=product_mentioned,
                )

                await send_meta_reply(sender_id, reply)

    except Exception as e:
        print(f"[server log] Meta webhook processing failed: {e}")

    return {"status": "event_received"}


async def send_meta_reply(recipient_id: str, text_message: str):
    if not META_PAGE_ACCESS_TOKEN:
        print("[server log] META_PAGE_ACCESS_TOKEN not set - cannot send reply.")
        return

    # graph.instagram.com, NOT graph.facebook.com - tokens from the
    # standalone Instagram Login API (starting with "IGAA") only work
    # against Instagram's own Graph API host, confirmed by testing.
    url = f"https://graph.instagram.com/{META_GRAPH_API_VERSION}/me/messages"
    params = {"access_token": META_PAGE_ACCESS_TOKEN}
    body = {
        "recipient": {"id": recipient_id},
        "message": {"text": text_message},
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, params=params, json=body)
        print(f"[server log] Meta send API response: {response.status_code} {response.text}")


# In-memory history resets on restart, which is acceptable for testing.
#
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN")
WHATSAPP_ACCESS_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_GRAPH_API_VERSION = "v21.0"

whatsapp_conversation_memory: dict[str, list[dict]] = {}


@app.get("/webhooks/whatsapp")
def verify_whatsapp_webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        print("[server log] WhatsApp webhook verified successfully.")
        return Response(content=challenge, media_type="text/plain")

    raise HTTPException(status_code=403, detail="Verification token mismatch")


@app.post("/webhooks/whatsapp")
async def handle_whatsapp_message(request: Request):
    payload = await request.json()

    try:
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                for message in value.get("messages", []):
                    sender_id = message.get("from")
                    message_text = message.get("text", {}).get("body")

                    if not sender_id or not message_text:
                        continue

                    print(f"[server log] WhatsApp message from {sender_id}: {message_text}")

                    history = whatsapp_conversation_memory.get(sender_id, [])

                    (reply, was_clarification, search_query, matched_products, where_filter,
                     best_score, category, customer_name, customer_phone, product_mentioned) = get_bot_response(
                        message_text, history
                    )

                    history.append({"role": "user", "content": message_text})
                    history.append({"role": "assistant", "content": reply})
                    whatsapp_conversation_memory[sender_id] = history[-12:]

                    log_conversation(
                        session_id=sender_id,
                        channel="whatsapp",
                        category=category,
                        user_message=message_text,
                        search_query_used=search_query,
                        bot_reply=reply,
                        matched_products=matched_products,
                        where_filter=where_filter,
                        best_score=best_score,
                        was_clarification=was_clarification,
                        customer_name=customer_name,
                        customer_phone=customer_phone,
                        product_mentioned=product_mentioned,
                    )

                    await send_whatsapp_reply(sender_id, reply)

    except Exception as e:
        print(f"[server log] WhatsApp webhook processing failed: {e}")

    return {"status": "event_received"}


async def send_whatsapp_reply(recipient_id: str, text_message: str):
    if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        print("[server log] WHATSAPP_ACCESS_TOKEN or WHATSAPP_PHONE_NUMBER_ID not set - cannot send reply.")
        return

    url = f"https://graph.facebook.com/{WHATSAPP_GRAPH_API_VERSION}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}
    body = {
        "messaging_product": "whatsapp",
        "to": recipient_id,
        "type": "text",
        "text": {"body": text_message},
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=body)
        print(f"[server log] WhatsApp send API response: {response.status_code} {response.text}")