import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
from heapq import nlargest

# Load the spaCy model once
@st.cache_resource
def load_spacy_model():
    nlp = spacy.load('en_core_web_sm')
    punctuations = string.punctuation + '\n'
    return nlp, list(STOP_WORDS), punctuations

nlp, stopwords, punctuations = load_spacy_model()

def summarize_text_extractive(text, percentage=0.3):
    if not text or percentage <= 0:
        return ""

    doc = nlp(text)

    # 1. Word Frequency Calculation (Cleaning and Tokenization implicit)
    word_frequencies = {}
    # Filter out stop words and punctuation, then count frequencies
    for token in doc:
        word = token.text.lower().strip()
        if word not in stopwords and word not in punctuations and word.isalpha():
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    if not word_frequencies:
        return "Not enough meaningful words to summarize."

    # 2. Normalize Word Count
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    # 3. Sentence Tokenization
    mysentences = [sent for sent in doc.sents]

    # 4. Calculate Sentence Frequencies (Importance Score)
    sentences_frequencies = {}
    for sent in mysentences:
        score = 0
        for word in sent:
            word_lower = word.text.lower()
            if word_lower in word_frequencies.keys():
                score += word_frequencies[word_lower]
        sentences_frequencies[sent] = score

    # 5. Select N-largest Sentences
    total_sentences = len(mysentences)
    # Calculate number of sentences to select (at least 1, max total_sentences)
    num_sent = max(1, min(total_sentences, int(total_sentences * percentage)))

    # Use nlargest to get the top 'num_sent' sentences based on their score
    summary_sentences = nlargest(num_sent, sentences_frequencies, key=sentences_frequencies.get)

    # 6. Combine to make the final summary
    final_summary = ' '.join([sent.text for sent in summary_sentences])

    return final_summary

# --- Streamlit UI Section ---
st.title("NLP Text Summarizer")
st.markdown("An extractive summarization tool built using spaCy and word frequency scoring.")

# Text Input Area
text_input = st.text_area(
    "Paste your text here:",
    height=250,
    placeholder="Enter the text you want to summarize..."
)

# Summarization percentage slider
percentage = st.slider(
    "Select Summary Length (% of original sentences):",
    min_value=10,
    max_value=90,
    value=30,
    step=10
) / 100  # Convert percentage to a decimal for the function

if st.button("Generate Summary"):
    if text_input:
        with st.spinner('Summarizing text...'):
            summary = summarize_text_extractive(text_input, percentage)

        st.success("Summary Generated!")
        st.subheader("Final Summary:")
        st.info(summary)

        # Optional: Display stats
        st.markdown("---")
        st.subheader("Analysis")
        original_length = len(text_input.split())
        summary_length = len(summary.split())
        st.write(f"Original Text Length: **{original_length} words**")
        st.write(f"Summary Length: **{summary_length} words**")
        st.write(f"Reduction: **{100 - (summary_length/original_length) * 100:.2f}%**")

    else:
        st.warning("Please paste some text to summarize.")