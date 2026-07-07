import streamlit as st
import pickle
import re
import nltk

# Download required NLTK data (only runs once)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
@st.cache_resource
def load_resources():
    try:
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        with open("logistic_regression_model.pkl", "rb") as f:
            model = pickle.load(f)

        return vectorizer, model

    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        st.stop()

tfidf_vectorizer, logistic_model = load_resources()

# -----------------------------
# Preprocessing
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and non-alpha tokens
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(tokens)

# -----------------------------
# UI
# -----------------------------
st.title("🎬 IMDB Movie Review Sentiment Analyzer")

st.write(
    "Enter a movie review below and the model will predict "
    "**Positive 😊** or **Negative 😞** sentiment."
)

review = st.text_area(
    "Movie Review",
    height=200,
    placeholder="Example: This movie was amazing! I loved every minute of it."
)

if st.button("Predict Sentiment 🚀"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
        st.stop()

    with st.spinner("Analyzing..."):

        processed = preprocess_text(review)

        vector = tfidf_vectorizer.transform([processed])

        prediction = logistic_model.predict(vector)[0]

        probabilities = logistic_model.predict_proba(vector)[0]
        confidence = probabilities.max()

    st.divider()

    st.subheader("Prediction")

    # Works whether labels are strings or integers
    if str(prediction).lower() in ["positive", "1"]:
        st.success(f"😊 Positive Review\n\nConfidence: {confidence:.2%}")
        st.balloons()
    else:
        st.error(f"😞 Negative Review\n\nConfidence: {confidence:.2%}")

st.divider()

st.caption(
    "Built using Logistic Regression + TF-IDF Vectorizer trained on the IMDB Movie Review Dataset."
)
