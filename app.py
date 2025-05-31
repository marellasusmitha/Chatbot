import os
import nltk
import ssl
import random
import time
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# NLTK setup
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Sample intents with emojis
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hey there! 😊", "Hi hi! 👋", "What's up? 😎", "I'm vibin' ✨ How about you?"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["See ya! 👋", "Take care! 💖", "Catch you later! ✌️"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome! 💫", "No worries 😄", "Glad to help! 🤗"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I'm your friendly lil' chatbot 🤖", "Here to chat, chill, and vibe with you 💬✨"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Of course! 💡 Just ask me anything.", "I got you! 💪 What do you need?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I'm forever young 🧸", "Age is just a number when you're a chatbot 😉"]
    }
]

# Train model
corpus = []
tags = []
for intent in intents:
    for pattern in intent['patterns']:
        corpus.append(pattern)
        tags.append(intent['tag'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
model = LogisticRegression()
model.fit(X, tags)

def get_response(user_input):
    X_test = vectorizer.transform([user_input])
    prediction = model.predict(X_test)[0]
    for intent in intents:
        if intent["tag"] == prediction:
            return random.choice(intent["responses"])
    return "Hmm... I didn't get that 😅 Try again?"

# Streamlit page setup
st.set_page_config(page_title="Casual Chatbot", layout="centered")

# Custom CSS for clean responsive look
st.markdown("""
<style>
.chat-container {
    max-width: 700px;
    margin: auto;
    padding: 20px;
}
.message {
    padding: 10px 15px;
    margin: 8px 0;
    border-radius: 20px;
    max-width: 75%;
    word-wrap: break-word;
}
.user {
    background-color: #d1f3ff;
    margin-left: auto;
    text-align: right;
}
.bot {
    background-color: #ffd6dc;
    margin-right: auto;
    text-align: left;
}
.typing {
    font-style: italic;
    color: #999;
    margin: 5px 0;
}
.input-box {
    width: 100%;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# App title
st.title("💬 Casual Chatbot")

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Session state to store conversation
if "chat" not in st.session_state:
    st.session_state.chat = []

# Text input box
user_input = st.text_input("Type your message here...", label_visibility="collapsed", placeholder="Start chatting...")

# Process user input
if user_input:
    st.session_state.chat.append(("user", user_input))
    with st.spinner("Typing..."):
        time.sleep(1.2)  # Simulated typing
    response = get_response(user_input)
    st.session_state.chat.append(("bot", response))

# Display chat history
for sender, message in st.session_state.chat:
    role = "user" if sender == "user" else "bot"
    st.markdown(f'<div class="message {role}">{message}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close chat-container
