# app.py
import streamlit as st
import pickle
from corrections import detect_errors, display_tokens, FUNCTION_WORDS
from nltk.metrics import edit_distance

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Spelling & Grammar Correction System",
    page_icon="✍️",
    layout="centered"
)

# -----------------------------
# Load precomputed data
# -----------------------------
@st.cache_resource
def load_models():
    with open("word_freq.pkl", "rb") as f:
        word_freq = pickle.load(f)
    vocab = set(word_freq.keys())

    with open("bigram_counts.pkl", "rb") as f:
        bigram_counts = pickle.load(f)

    with open("unigram_counts.pkl", "rb") as f:
        unigram_counts = pickle.load(f)

    return word_freq, vocab, bigram_counts, unigram_counts

WORD_FREQ, VOCAB, BIGRAM_COUNTS, UNIGRAM_COUNTS = load_models()

# -----------------------------
# Sidebar: Vocabulary Explorer
# -----------------------------
st.sidebar.title("📚 Corpus Vocabulary")
search_word = st.sidebar.text_input("Search for a word:")
sorted_vocab = sorted(VOCAB)

if search_word:
    filtered_vocab = [w for w in sorted_vocab if search_word.lower() in w]
else:
    filtered_vocab = sorted_vocab

st.sidebar.write(f"Showing {len(filtered_vocab)} words")
st.sidebar.dataframe(filtered_vocab[:1000], height=400)

# -----------------------------
# UI Header
# -----------------------------
st.title("✍️ Spelling & Grammar Correction System")
st.markdown("""
This application detects:
- **Spelling errors** (red)
- **Grammar-aware corrections** (green, e.g., "is rise → is rising")
Click on a highlighted word to see suggested corrections (sorted by minimum edit distance).
""")
st.divider()

# -----------------------------
# User Input
# -----------------------------
st.subheader("📝 Enter Text")
user_input = st.text_area(
    "Type or paste your text below (maximum 500 characters):",
    height=180,
    max_chars=500,
    placeholder="Example: Bitcoin is rise this year"
)

# -----------------------------
# Spell & Grammar Check
# -----------------------------
if st.button("🔍 Check Text", use_container_width=True):

    if not user_input.strip():
        st.warning("⚠️ Please enter some text before checking.")
    else:
        # 1️⃣ Grammar-aware display tokens + grammar indices
        display_version, grammar_indices = display_tokens(user_input)

        # 2️⃣ Detect spelling errors
        errors = detect_errors(user_input)
        spelling_words = {err['word'] for err in errors}  # red highlights

        # Map word → suggestions sorted by edit distance
        suggestion_map = {}
        for err in errors:
            suggestions = sorted(
                err.get('suggestions', []),
                key=lambda w: edit_distance(err['word'], w)
            )
            suggestion_map[err['word']] = suggestions

        # 3️⃣ Build highlighted text
        highlighted_text = []
        for i, word in enumerate(display_version):
            lw = word.lower()
            if lw in spelling_words and lw not in FUNCTION_WORDS:
                highlighted_text.append(f"[**:red[{word}]**](#)")  # spelling error
            elif i in grammar_indices:
                highlighted_text.append(f"[**:green[{word}]**](#)")  # grammar correction
            else:
                highlighted_text.append(word)

        st.subheader("🖍 Highlighted Text")
        st.markdown(" ".join(highlighted_text))

        # -----------------------------
        # Error Details & Suggestions
        # -----------------------------
        st.divider()
        st.subheader("📌 Error Details & Suggestions")

        if not errors:
            st.success("✅ No spelling errors detected!")
        else:
            for i, word in enumerate(display_version):
                lw = word.lower()
                if lw in spelling_words or i in grammar_indices:
                    suggestions = suggestion_map.get(lw, [])
                    err_type = next((err['type'] for err in errors if err['word']==lw), "grammar")
                    with st.expander(f"❌ `{word}` — {err_type} error"):
                        if suggestions:
                            st.markdown("**Suggested corrections (sorted by edit distance):**")
                            for s in suggestions:
                                dist = edit_distance(lw, s)
                                st.markdown(f"- **{s}** (Edit distance: {dist})")
                        else:
                            st.info("No suitable suggestions found.")

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("📘 MSc Artificial Intelligence | Spelling & Grammar Correction")
