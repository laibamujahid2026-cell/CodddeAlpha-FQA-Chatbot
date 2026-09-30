# FAQ Chatbot

CodeAlpha Artificial Intelligence Internship — Task 2

## What it does

A chat-style web app where a user types a question and the bot replies
with the most relevant answer from a set of prepared FAQs.

## How it works

1. **Collecting FAQs** — Questions and answers are stored in
   `faqs.json`, so they're easy to edit or extend.
2. **Preprocessing** — Every question (from the FAQ list and from the
   user) is lowercased, stripped of punctuation, tokenized, cleaned of
   stopwords, and lemmatized using **NLTK**.
3. **Matching** — The cleaned questions are converted into TF-IDF
   vectors with **scikit-learn**, and the user's question is compared
   against every FAQ question using **cosine similarity**. The FAQ
   with the highest similarity score is chosen as the answer.
4. **Chat UI** — Built with Streamlit's `st.chat_message` /
   `st.chat_input`, so it behaves like a normal chat window and keeps
   the conversation history for the session.

If no FAQ is similar enough (score below a threshold), the bot admits
it isn't sure instead of guessing.

## Project structure

```
CodeAlpha_FAQChatbot/
├── app.py            # Main application code
├── faqs.json          # The FAQ knowledge base (edit this to add more Q&A)
├── requirements.txt  # Python dependencies
└── README.md          # This file
```

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually
`http://localhost:8501`) and start chatting.

## Possible improvements

- Swap TF-IDF for sentence embeddings for smarter matching.
- Let users rate whether an answer was helpful.
- Add more FAQs, or load them from a spreadsheet/CMS.
