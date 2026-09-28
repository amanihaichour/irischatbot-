from api import classify_message

test_cases = [
    ("merci beaucoup, vous êtes super!", "remerciement"),
    ("c'est un scandale, ma commande est cassée", "reclamation"),
    ("est-ce que vous vendez des télévisions 4K?", "demande_info"),
    ("combien coûte ce frigo?", "demande_prix"),
    ("où en est ma commande #4521, elle devait arriver hier", "livraison_commande"),
    ("bonjour, comment allez-vous?", "autre"),
]

correct = 0
for message, expected in test_cases:
    got = classify_message(message)
    status = "OK" if got == expected else "MISMATCH"
    if got == expected:
        correct += 1
    print(f"[{status}] expected={expected:20} got={got:20} msg={message!r}")

print(f"\n{correct}/{len(test_cases)} correct")