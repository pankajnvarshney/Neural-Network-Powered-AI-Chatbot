
import streamlit as st
import numpy as np
import random
import joblib
import json
from keras.models import load_model

# Load assets
model = load_model("chatbot_model.h5")
vectorizer = joblib.load("vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

with open("intents.json") as file:
    intents = json.load(file)

def predict_class(text):
    vec = vectorizer.transform([text]).toarray()
    pred = model.predict(vec)[0]
    tag_idx = np.argmax(pred)
    tag = encoder.inverse_transform([tag_idx])[0]
    return tag

def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

# Streamlit UI
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot")
st.markdown("Ask me anything!")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", key="input")

if st.button("Send"):
    if user_input:
        tag = predict_class(user_input)
        bot_reply = get_response(tag)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", bot_reply))

# Display chat history
for sender, msg in st.session_state.history:
    if sender == "You":
        st.markdown(f"**🧑‍💻 {sender}:** {msg}")
    else:
        st.markdown(f"**🤖 {sender}:** {msg}")
