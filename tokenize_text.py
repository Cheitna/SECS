# tokenize_text.py
"""
Stage 2: Tokenization & Lemmatization (for corpus)
- Split text into words
- Remove punctuation and numbers
- Preserve auxiliary verbs for context
- Lemmatize content verbs using WordNet
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download NLTK resources (run once)
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

lemmatizer = WordNetLemmatizer()

# Stopwords
STOPWORDS = set(stopwords.words("english"))
KEEP = {"is", "was", "are", "this", "that"}  # preserve for context
STOPWORDS = STOPWORDS - KEEP

# Auxiliary verbs to preserve
AUX_VERBS = {"am", "is", "are", "was", "were"}

# Read cleaned text
with open("cleaned.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize: only alphabetic words
tokens = re.findall(r'\b[a-z]+\b', text)

# POS tagging
tagged_tokens = pos_tag(tokens)

processed_tokens = []

for word, tag in tagged_tokens:
    # Skip stopwords (except AUX_VERBS and KEEP)
    if word in STOPWORDS:
        continue

    # Preserve auxiliary verbs as-is
    if word in AUX_VERBS:
        lemma = word
    # Lemmatize content verbs
    elif tag.startswith("V"):
        lemma = lemmatizer.lemmatize(word, pos="v")
    # Lemmatize other words
    else:
        lemma = lemmatizer.lemmatize(word)

    processed_tokens.append(lemma)

# Save tokens for vocabulary & bigram building
with open("tokens.txt", "w", encoding="utf-8") as f:
    for token in processed_tokens:
        f.write(token + "\n")

print(f"Stage 2: Tokenization complete. Total tokens: {len(processed_tokens)}")
