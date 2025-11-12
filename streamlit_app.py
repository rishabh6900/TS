import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
from heapq import nlargest
import os
import sys

# Load the spaCy model once and cache the result for performance
@st.cache_resource
def load_spacy_model():
    """
    Attempts to load the spaCy model 'en_core_web_sm'.
    Includes a robust fallback for deployment environments where the model
    might not be in spaCy's default search path.
    """
    model_name = 'en_core_web_sm'
    try:
        # 1. Attempt standard loading (most efficient if paths are correct)
        nlp = spacy.load(model_name)
    except OSError:
        # 2. Fallback: Try to find the model in the site-packages directory
        # This is a common location in isolated environments like Streamlit Cloud
        
        # Determine the base path for site-packages/lib
        # For Linux deployments, this is often the venv's lib directory
        base_path = os.path.join(sys.prefix, 'lib')
        
        # Check Python version to construct the correct path
        python_version_dir = f'python{sys.version_info.major}.{sys.version_info.minor}'
        
        # Construct the expected path to the model package directory
        model_path = os.path.join(base_path, python_version_dir, 'site-packages', model_name)
        
        # Check for alternative model paths which might be slightly different 
        # based on the OS/environment setup.
        if not os.path.isdir(model_path):
            # Try a path for direct installation (no python version dir)
             model_path = os.path.join(sys.prefix, 'lib', 'site-packages', model_name)

        if os.path.isdir(model_path):
            try:
                # Load using the explicit path
                nlp = spacy.load(model_path)
                st.info(f"Successfully loaded spaCy model from custom path: {model_path}")
            except Exception as path_e:
                raise Exception(f"Failed to load model from both standard location and custom path: {path_e}")
        else:
            # If the path is not found, raise the original error for debugging
            raise

    # Combine standard punctuation with newline character
    punctuations = string.punctuation + '\n'
    return nlp, list(STOP_WORDS), punctuations

# Initialize global resources
try:
    # Remove the old import error check here, as the loading function now handles it
    nlp, stopwords, punctuations = load_spacy_model()
except Exception as e:
    # Catch any error from load_spacy_model, including the final failure
    st.error(
        f"Fatal Error: Could not load the spaCy model 'en_core_web_sm'. "
        "Please check your deployment logs and ensure the model is installed."
    )
    # Display the specific error message for better debugging
    st.exception(e) 
    st.stop()


def summarize_text_extractive(text, percentage=0.3):
    """
    Performs extractive text summarization based on word frequency scoring.
    """
    if not text or percentage <= 0:
        return ""

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
        # This check is technically redundant due to st.stop() above, but is a safe guard
        if 'nlp' not in globals() or nlp is None:
            st.warning("Model failed to load during startup. Cannot process request.")
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

