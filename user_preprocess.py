# user_preprocess.py
"""
Grammar-aware preprocessing for user input
- Lowercase
- Tokenize
- Lemmatize using WordNet
- Apply simple grammar rules:
    1. is/are/am/was/were + VB -> VBG (present participle)
    2. has/have/had + (optional 'not') + VB -> VBN (past participle)
- Returns:
    - lemma tokens (for detection)
    - display tokens (with grammar corrections applied)
    - indices of grammar-corrected tokens
"""

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download resources
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()

# Auxiliary verbs
AUX_VERBS = {"am", "is", "are", "was", "were", "has", "have", "had"}

# Present participle for BE verbs
BASE_VERB_TO_ING = {
    "be": "being",
    "have": "having",
    "do": "doing",
    "go": "going",
    "rise": "rising",
    "come": "coming",
    "happen": "happening",
    "move": "moving",
    "determine": "determining",
    "use": "using"
}

# Past participles for HAS/HAVE/HAD
IRREGULAR_PAST = {
    "move": "moved",
    "go": "gone",
    "come": "come",
    "rise": "risen",
    "be": "been",
    "have": "had",
    "do": "done"
}

# Function words: never flagged as errors
FUNCTION_WORDS = {"this", "that", "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with", "by", "at"}

# -----------------------------
# Preprocessing for detection
# -----------------------------
def preprocess_user_input(text):
    """
    Lemmatize tokens for spelling detection.
    """
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    tagged_tokens = pos_tag(tokens)

    processed = []
    for word, tag in tagged_tokens:
        # Keep auxiliary verbs as-is
        if word in AUX_VERBS:
            lemma = word
        # Lemmatize verbs
        elif tag.startswith("V"):
            lemma = lemmatizer.lemmatize(word, pos="v")
        else:
            lemma = lemmatizer.lemmatize(word)
        processed.append(lemma)
    return processed

# -----------------------------
# Grammar correction for display
# -----------------------------
def apply_display_grammar(tokens):
    """
    Convert lemma tokens into display-friendly form.
    Handles:
    - BE + VB -> VBG
    - HAS/HAVE/HAD + (optional 'not') + VB -> VBN
    Returns:
        display_tokens: tokens with grammar applied
        grammar_indices: indices of corrected tokens (for green highlight)
    """
    display_tokens = []
    grammar_indices = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]
        prev = display_tokens[-1] if i > 0 else ""

        # -----------------------
        # Present participle rule: is/are/am/was/were + VB
        # -----------------------
        if prev in {"am", "is", "are", "was", "were"} and tok in BASE_VERB_TO_ING:
            display_tokens.append(BASE_VERB_TO_ING[tok])
            grammar_indices.append(i)
            i += 1
            continue

        # -----------------------
        # Past participle rule: has/have/had + optional 'not' + VB
        # -----------------------
        if prev in {"has", "have", "had"}:
            # Check if next token is negation
            if tok == "not" and (i + 1 < len(tokens)):
                next_tok = tokens[i + 1]
                past_tok = IRREGULAR_PAST.get(next_tok, next_tok)  # correct past
                display_tokens.append(tok)  # keep 'not'
                display_tokens.append(past_tok)
                grammar_indices.append(i + 1)
                i += 2
                continue
            else:
                past_tok = IRREGULAR_PAST.get(tok, tok)  # correct past
                display_tokens.append(past_tok)
                grammar_indices.append(i)
                i += 1
                continue

        # -----------------------
        # Default: no change
        # -----------------------
        display_tokens.append(tok)
        i += 1

    return display_tokens, grammar_indices

# -----------------------------
# Example test
# -----------------------------
if __name__ == "__main__":
    sentences = [
        "Bitcoin is rise this year",
        "62% of Bitcoin has not move in a year",
        "She has go to the market",
        "He is come late"
    ]

    for sent in sentences:
        lemma_tokens = preprocess_user_input(sent)
        display_tokens, grammar_idx = apply_display_grammar(lemma_tokens)
        print("Input:", sent)
        print("Lemma tokens:", lemma_tokens)
        print("Display tokens:", display_tokens)
        print("Grammar corrected indices:", grammar_idx)
        print("-" * 50)
