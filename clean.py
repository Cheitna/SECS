# clean.py
"""
Stage 1: Clean the raw dataset.
- Lowercase text
- Remove extra whitespace
- Save cleaned text
"""

with open("data2.txt", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# Lowercase
cleaned_text = text.lower()

# Remove extra whitespace
cleaned_text = " ".join(cleaned_text.split())

# Save cleaned text
with open("cleaned.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print("Stage 1: Cleaning complete.")
