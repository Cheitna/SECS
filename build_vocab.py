# build_vocab.py
"""
Stage 3: Build vocabulary & word frequency
- Count token frequencies
- Save unique vocabulary
- Save frequency dictionary (for candidate ranking)
"""

from collections import Counter
import pickle

# Read tokens
with open("tokens.txt", "r", encoding="utf-8") as f:
    tokens = f.read().splitlines()

# Count frequencies
word_freq = Counter(tokens)

# Optional: remove extremely rare words (threshold >=2)
vocab = {word for word, freq in word_freq.items() if freq >= 2}

# Save vocabulary
with open("vocabulary.txt", "w", encoding="utf-8") as f:
    for word in sorted(vocab):
        f.write(word + "\n")

# Save word frequency dictionary
with open("word_freq.pkl", "wb") as f:
    pickle.dump(word_freq, f)

print(f"Stage 3: Vocabulary built. Unique words: {len(vocab)}")
