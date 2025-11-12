import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
from heapq import nlargest

# Load the spaCy model once and cache the result for performance
# and to avoid re-loading the large model on every user interaction.
@st.cache_resource
def load_spacy_model():
    """Load the language model and create a set of punctuations."""
    # Use the Python package name directly. This is the most reliable way 
    # when the model is installed via requirements.txt
    import en_core_web_sm 
    nlp = en_core_web_sm.load()
    
    # Combine standard punctuation with newline character
    punctuations = string.punctuation + '\n'
    return nlp, list(STOP_WORDS), punctuations

# Initialize global resources
try:
    nlp, stopwords, punctuations = load_spacy_model()
except Exception as e:
    st.error(
        f"An error occurred during model loading. Ensure spaCy and its model are installed correctly. Error details: {e}"
    )
    # Use st.stop() to prevent the rest of the script from executing if the model load fails
    st.stop()


def summarize_text_extractive(text, percentage=0.3):
    """
    Performs extractive text summarization based on word frequency scoring.
    """
    if not text or percentage <= 0:
        return ""

    # doc = nlp(text) is already using the globally loaded nlp object
    doc = nlp(text)

    # 1. Word Frequency Calculation (Cleaning and Tokenization implicit)
    word_frequencies = {}
    for token in doc:
        word = token.text.lower().strip()
        # Filter out stop words, punctuation, and non-alphabetic tokens
        if word not in stopwords and word not in punctuations and word.isalpha():
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    if not word_frequencies:
        return "Not enough meaningful words to summarize."

    # 2. Normalize Word Count (to prevent large differences in scores)
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
        # Map the spaCy sentence object to its score
        sentences_frequencies[sent] = score

    # 5. Select N-largest Sentences
    total_sentences = len(mysentences)
    # Calculate number of sentences to select (ensuring at least 1)
    num_sent = max(1, min(total_sentences, int(total_sentences * percentage)))

    # Use nlargest to get the top 'num_sent' sentences based on score
    summary_sentences = nlargest(num_sent, sentences_frequencies, key=sentences_frequencies.get)

    # 6. Combine to make the final summary
    final_summary = ' '.join([sent.text for sent in summary_sentences])

    return final_summary

# --- Streamlit UI Section ---
st.set_page_config(page_title="Text Summarizer", layout="wide")

st.title("🤖 Extractive Text Summarizer")
st.markdown("Use this tool to generate concise summaries from long articles based on word frequency scoring (spaCy).")

# Text Input Area
text_input = st.text_area(
    "Paste your text here:",
    height=300,
    placeholder="Enter the text you want to summarize (e.g., an article, report, or document)."
)

# Summarization percentage slider
percentage = st.slider(
    "Select Summary Length (% of original sentences):",
    min_value=10,
    max_value=90,
    value=30,
    step=10,
    help="This percentage determines how many of the original sentences will be included in the summary."
) / 100

if st.button("Generate Summary", type="primary"):
    if text_input:
        # Check if nlp object is available before processing
        if 'nlp' not in globals() or nlp is None:
            st.warning("Model failed to load. Please check installation logs.")
            st.stop()
            
        with st.spinner('Summarizing text... This may take a moment for large documents.'):
            summary = summarize_text_extractive(text_input, percentage)

        st.success("Summary Generated!")

        st.markdown("### Final Summary")
        st.info(summary)

        # Optional: Display stats
        st.markdown("---")
        st.subheader("Summary Analysis")
        original_words = len(text_input.split())
        summary_words = len(summary.split())
        
        # Avoid division by zero if original text is empty or too short
        if original_words > 0:
            reduction_percent = 100 - (summary_words / original_words) * 100
        else:
            reduction_percent = 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Word Count", f"{original_words}")
        with col2:
            st.metric("Summary Word Count", f"{summary_words}")
        with col3:
            st.metric("Reduction", f"{reduction_percent:.2f}%")

    else:
        st.warning("Please paste some text into the box above and try again.")
