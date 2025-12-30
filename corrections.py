# corrections.py
"""
Spelling & Grammar Correction Module
- Non-word detection (edit distance + frequency)
- Real-word detection (bigram probability with Laplace smoothing)
- Grammar-aware display (e.g., is + rise → is rising)
"""

import pickle
from nltk.metrics.distance import edit_distance
from user_preprocess import preprocess_user_input, apply_display_grammar

# -----------------------------
# Extended function words (never flagged)
# -----------------------------
FUNCTION_WORDS = {
    # Articles, determiners, prepositions
    "this","that","a","an","the","and","or","of","in","on","for","to","with","by","at",
    # Pronouns
    "i","you","he","she","it","we","they","me","him","her","us","them",
    # Auxiliaries
    "am","is","are","was","were","has","have","had","do","does","did","will","would","can","could","shall","should","may","might","must",
    # Conjunctions
    "if","but","because","so","than","as","while","although","though"
}

# -----------------------------
# Load precomputed models
# -----------------------------
with open("vocabulary.txt", "r", encoding="utf-8") as f:
    VOCAB = set(w.lower() for w in f.read().splitlines()) | FUNCTION_WORDS # ensure lowercase

with open("word_freq.pkl", "rb") as f:
    WORD_FREQ = pickle.load(f)

with open("bigram_counts.pkl", "rb") as f:
    BIGRAM_COUNTS = pickle.load(f)

with open("unigram_counts.pkl", "rb") as f:
    UNIGRAM_COUNTS = pickle.load(f)

TOTAL_UNIGRAMS = sum(UNIGRAM_COUNTS.values())
VOCAB_SIZE = len(VOCAB)  # for Laplace smoothing

# -----------------------------
# Bigram probability with Laplace smoothing
# -----------------------------
def bigram_prob_laplace(w1, w2):
    """Returns P(w2 | w1) with add-one (Laplace) smoothing"""
    w1 = w1.lower()
    w2 = w2.lower()
    count_bigram = BIGRAM_COUNTS.get((w1, w2), 0)
    count_unigram = UNIGRAM_COUNTS.get(w1, 0)
    return (count_bigram + 1) / (count_unigram + VOCAB_SIZE)

# -----------------------------
# Candidate generation and ranking
# -----------------------------
def generate_candidates(word, max_distance=2):
    """Generate candidates from VOCAB within edit distance threshold"""
    word = word.lower()
    return [w for w in VOCAB if edit_distance(word, w) <= max_distance]

def rank_candidates(candidates, prev_word=None):
    """
    Rank candidates using:
    1. Bigram probability (if previous word given)
    2. Word frequency
    """
    ranked = []
    for cand in candidates:
        score = WORD_FREQ.get(cand, 0) / TOTAL_UNIGRAMS  # frequency component
        if prev_word and prev_word.lower() not in FUNCTION_WORDS:
            score += bigram_prob_laplace(prev_word, cand)
        ranked.append((cand, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:5]]  # top 5 suggestions

# -----------------------------
# Main error detection
# -----------------------------
def detect_errors(user_text):
    """
    Detect non-word and real-word errors
    Returns a list of dicts: {'word', 'type', 'suggestions'}
    """
    tokens = preprocess_user_input(user_text)
    errors = []

    for i, token in enumerate(tokens):
        token_lc = token.lower()
        prev_word = tokens[i-1] if i > 0 else None

        # Skip function words
        if token_lc in FUNCTION_WORDS:
            continue

        # Non-word error
        if token_lc not in VOCAB:
            candidates = generate_candidates(token_lc)
            errors.append({
                'word': token,
                'type': 'non-word',
                'suggestions': rank_candidates(candidates, prev_word)
            })
        else:
            # Real-word error (contextually unlikely)
            if prev_word and prev_word.lower() not in FUNCTION_WORDS:
                prob = bigram_prob_laplace(prev_word, token_lc)
                if prob < 1e-6:  # adjust threshold based on corpus
                    candidates = generate_candidates(token_lc)
                    errors.append({
                        'word': token,
                        'type': 'real-word',
                        'suggestions': rank_candidates(candidates, prev_word)
                    })

    return errors

# -----------------------------
# Display-friendly tokens
# -----------------------------
def display_tokens(user_text):
    """
    Returns tokens for display along with grammar-corrected indices
    Example: is + rise → is rising
    """
    lemmas = preprocess_user_input(user_text)
    display_version, grammar_indices = apply_display_grammar(lemmas)
    return display_version, grammar_indices

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    test_sentence = "AI helps in mny field. The technique automatically determines which optimization algorithm it should use."
    display, grammar_idxs = display_tokens(test_sentence)
    errors = detect_errors(test_sentence)
    print("Display:", display)
    print("Grammar indices:", grammar_idxs)
    for err in errors:
        print(f"Word: {err['word']} | Type: {err['type']} | Suggestions: {err['suggestions']}")
