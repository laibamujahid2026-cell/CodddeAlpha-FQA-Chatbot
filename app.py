"""
FAQ Chatbot
-----------
A simple chatbot that answers user questions by matching them against
a bank of frequently asked questions (FAQs). It preprocesses text with
NLTK, then finds the closest FAQ using TF-IDF + cosine similarity.

Run locally with:
    streamlit run app.py
"""

import json
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------
# One-time NLTK downloads (cached so this only runs once per environment)
# ---------------------------------------------------------------------
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(
            f"tokenizers/{resource}"
            if "punkt" in resource
            else f"corpora/{resource}"
        )
    except LookupError:
        nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    """Lowercase, strip punctuation, remove stopwords, and lemmatize."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    cleaned = [
        lemmatizer.lemmatize(word) for word in tokens if word not in STOP_WORDS
    ]
    return " ".join(cleaned)


# ---------------------------------------------------------------------
# Load FAQs and build the similarity model
# ---------------------------------------------------------------------
@st.cache_resource
def load_faq_model():
    with open("faqs.json", "r", encoding="utf-8") as f:
        faqs = json.load(f)

    questions = [faq["question"] for faq in faqs]
    processed_questions = [preprocess(q) for q in questions]

    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(processed_questions)

    return faqs, vectorizer, question_vectors


faqs, vectorizer, question_vectors = load_faq_model()


def get_best_answer(user_question: str, threshold: float = 0.2):
    """Return the FAQ answer whose question is most similar to the
    user's question, or a fallback message if nothing matches well."""
    processed = preprocess(user_question)
    user_vector = vectorizer.transform([processed])

    similarities = cosine_similarity(user_vector, question_vectors)[0]
    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score < threshold:
        return (
            "I'm not sure about that one. Could you rephrase your question, "
            "or ask something related to the internship FAQs?",
            best_score,
        )

    return faqs[best_index]["answer"], best_score


# ---------------------------------------------------------------------
# Streamlit chat UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")
st.title("🤖 FAQ Chatbot")
st.write("Ask me a question about the internship and I'll do my best to answer.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show past messages
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    answer, score = get_best_answer(user_input)

    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(f"match confidence: {score:.2f}")

    st.session_state.chat_history.append(("assistant", answer))

with st.expander("See all FAQs this bot knows about"):
    for faq in faqs:
        st.markdown(f"**Q: {faq['question']}**")
        st.write(faq["answer"])
        st.write("---")

st.caption("Built for the CodeAlpha Artificial Intelligence internship — Task 2.")
