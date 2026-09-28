"""
analyze_logs.py

Reads data/chat_logs.db and produces:
  1. A CSV export (data/chat_logs_export.csv) - shareable, opens in Excel
  2. A printed summary: most common questions, clarification rate,
     lowest-confidence matches (= gaps in the product data worth
     knowing about), questions per session

This is the "Phase 6" piece of the original project plan: using
real customer questions to find what's missing.

Run with:
    python src/analyze_logs.py
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/chat_logs.db")
CSV_PATH = Path("data/chat_logs_export.csv")

if not DB_PATH.exists():
    print(f"No log file found at {DB_PATH} yet - chat with the bot first.")
    exit()

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM conversations ORDER BY timestamp", conn)
conn.close()

if df.empty:
    print("Log file exists but has no entries yet.")
    exit()

df.to_csv(CSV_PATH, index=False)
print(f"Exported {len(df)} conversations to {CSV_PATH}\n")

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total messages logged: {len(df)}")
print(f"Unique conversations (sessions): {df['session_id'].nunique()}")
print(f"Messages that triggered a clarification (bot was confused): "
      f"{df['was_clarification'].sum()} ({df['was_clarification'].mean()*100:.1f}%)")

if "category" in df.columns:
    print("\n" + "=" * 60)
    print("MESSAGES BY CATEGORY")
    print("=" * 60)
    category_counts = df["category"].value_counts()
    for category, count in category_counts.items():
        pct = count / len(df) * 100
        print(f"  {category}: {count} ({pct:.1f}%)")

print("\n" + "=" * 60)
print("LOWEST-CONFIDENCE MATCHES (worth reviewing - possible")
print("missing products, or questions your data can't answer)")
print("=" * 60)
worst = df.nlargest(10, "best_score")[["user_message", "search_query_used", "best_score", "was_clarification"]]
for _, row in worst.iterrows():
    print(f"  score={row['best_score']:.2f}  \"{row['user_message']}\"  (searched: \"{row['search_query_used']}\")")

print("\n" + "=" * 60)
print("MOST COMMON WORDS IN CUSTOMER QUESTIONS")
print("=" * 60)
all_words = " ".join(df["user_message"].str.lower()).split()
stopwords = {"le", "la", "les", "de", "des", "du", "un", "une", "et", "a", "the", "is",
             "do", "you", "have", "for", "que", "qui", "est", "vous", "avez"}
word_counts = {}
for w in all_words:
    w = w.strip("?!.,")
    if w and w not in stopwords and len(w) > 2:
        word_counts[w] = word_counts.get(w, 0) + 1

top_words = sorted(word_counts.items(), key=lambda x: -x[1])[:15]
for word, count in top_words:
    print(f"  {word}: {count}")

print(f"\nFull data available in {CSV_PATH} and {DB_PATH}")