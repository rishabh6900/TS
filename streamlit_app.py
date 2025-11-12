import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
from heapq import nlargest
import subprocess
import sys

# Define the model name globally
MODEL_NAME = 'en_core_web_sm'

# Load the spaCy model once and cache the result for performance
@st.cache_resource
def load_spacy_model():
    """
    Attempts to load the spaCy model. If it fails due to a missing file (E050),
    it executes a subprocess command to download and install the model into the
    current virtual environment during runtime.
    """
    try:
        # 1. Attempt standard loading
        nlp = spacy.load(MODEL_NAME)
        return nlp
    except OSError as e:
        # Check if the error is the specific E050 (Model not found)
        # Note: Streamlit redacts the E050 message, but if a load fails, this is the most likely cause.
        
        st.warning(f"spaCy model '{MODEL_NAME}' not found. Attempting to download and install it now...")
        
        # Determine the Python executable to ensure the model installs into the correct environment (venv)
        python_executable = sys.executable
        
        try:
            # Execute the download command as a subprocess.
            # We use '--no-warn' and '--quiet' to minimize console output noise during deployment.
            # We use 'run' to wait for the command to complete.
            result = subprocess.run([
                python_executable, 
                "-m", 
                "spacy", 
                "download", 
                MODEL_NAME, 
                "--no-warn",
                "--quiet"
            ], check=True, capture_output=True, text=True)
            
            # 2. Try loading again after successful download
            nlp = spacy.load(MODEL_NAME)
            st.success(f"Successfully downloaded and loaded model '{MODEL_NAME}' at runtime.")
            return nlp
            
        except subprocess.CalledProcessError as sub_e:
            st.error("Failed to download model via subprocess.")
            st.code(f"STDOUT: {sub_e.stdout}\nSTDERR: {sub_e.stderr}")
            raise Exception(f"Model download failed: {sub_e}")
        
        except Exception as retry_e:
            raise Exception(f"Model downloaded but failed to load: {retry_e}")

# Initialize global resources
try:
    # Load NLP object first
    nlp = load_spacy_model()
    
    # Initialize other variables after nlp object is secured
    punctuations = string.punctuation + '\n'
    stopwords = list(STOP_WORDS)
    
except Exception as e:
    # Catch any error from load_spacy_model, including the final failure
    st.error(
        f"Fatal Error: Could not initialize spaCy resources. "
        "Please check your deployment logs for model installation failures."
    )
    st.exception(e) 
    st.stop()
    
# Assign final variables outside the try/except if successful
nlp, stopwords, punctuations = nlp, stopwords, punctuations


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


