
import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Page Configuration ---
st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬")

# --- Load Model and Vectorizer ---
@st.cache_resource # Cache the model loading for performance
def load_resources():
    # Load the TF-IDF vectorizer
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as file:
            loaded_vectorizer = pickle.load(file)
    except FileNotFoundError:
        st.error("TF-IDF Vectorizer file not found! Please ensure 'tfidf_vectorizer.pkl' is in the same directory.")
        st.stop()

    # Load the Logistic Regression model
    try:
        with open('logistic_regression_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'logistic_regression_model.pkl' is in the same directory.")
        st.stop()
    return loaded_vectorizer, loaded_model

tfidf_vectorizer, logistic_model = load_resources()

# --- Text Preprocessing Functions (must match training preprocessing) ---
# Ensure NLTK data is available in the Streamlit environment if running standalone
# This part assumes NLTK data is already downloaded or will be handled by the environment setup.
# For Colab, we handled this in a previous cell.
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove non-alphabetic tokens and stopwords
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    return ' '.join(lemmatized_tokens)

# --- Streamlit UI ---
st.title('🎬 IMDB Movie Review Sentiment Analyzer')
st.markdown("Enter a movie review below, and I'll tell you if it's **positive** or **negative**!")

review_placeholder = "e.g., 'This movie was absolutely fantastic! The acting was superb and the story kept me on the edge of my seat.'"
review_input = st.text_area(
    'Your Movie Review:', 
    height=200, 
    placeholder=review_placeholder
)

if st.button('Predict Sentiment ✨'):
    if review_input:
        with st.spinner('Analyzing sentiment...'):
            # Preprocess the input review
            processed_review = preprocess_text(review_input)

            # Transform using the loaded TF-IDF vectorizer
            tfidf_review = tfidf_vectorizer.transform([processed_review])

            # Make prediction
            prediction = logistic_model.predict(tfidf_review)
            prediction_proba = logistic_model.predict_proba(tfidf_review)

            st.subheader('Prediction Result:')
            if prediction[0] == 'positive':
                st.success(f'✨ This review is **Positive**! (Confidence: {prediction_proba[0][1]:.2%})')
            else:
                st.error(f'💔 This review is **Negative**! (Confidence: {prediction_proba[0][0]:.2%})')

            # Optional: Show processed text (can be toggled for debugging)
            # with st.expander("See processed text and debug info"): # Example of expander for optional debug
            #     st.write(f'Original Review (first 100 chars): {review_input[:100]}...')
            #     st.write(f'Processed Review (first 100 chars): {processed_review[:100]}...')
            #     st.write(f'TF-IDF Shape: {tfidf_review.shape}')
    else:
        st.warning('Please enter some text in the review box to predict its sentiment.')

st.markdown("---")
st.info("This app uses a Logistic Regression model trained on the IMDB Movie Review Dataset.")
