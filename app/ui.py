import streamlit as st
import requests

st.set_page_config(page_title="Chat with Your Documents", page_icon="🧠", layout="wide")
st.title("🧠 Chat with Your Documents")
st.markdown("Ask questions about your PDF content.")

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.chat_turns = []

question = st.chat_input("Ask something about your documents...")
if question:
    resp = requests.post("http://localhost:8000/ask", json={
        "question": question,
        "history": st.session_state.chat_turns
    })
    answer = resp.json()["answer"]
    st.session_state.history.append((question, answer))
    st.session_state.chat_turns.append({"role": "user", "content": question})
    st.session_state.chat_turns.append({"role": "assistant", "content": answer})

for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)
