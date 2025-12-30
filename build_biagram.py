# build_bigram_model.py
from collections import Counter
import pickle

# Read preprocessed tokens
with open("tokens.txt", "r", encoding="utf-8") as f:
    tokens = f.read().splitlines()

# Build bigram counts
bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
bigram_counts = Counter(bigrams)

# Build unigram counts for probability calculation
unigram_counts = Counter(tokens)

# Save bigram counts
with open("bigram_counts.pkl", "wb") as f:
    pickle.dump(bigram_counts, f)

# Save unigram counts
with open("unigram_counts.pkl", "wb") as f:
    pickle.dump(unigram_counts, f)

print("Stage 4: Bigram model built.")
print(f"Unique bigrams: {len(bigram_counts)}")
