import streamlit as st
from agents import multy_ai_agent

st.title("📈 Finance AI Assistant")

question = st.text_input("Ask your Question about Finance")

if st.button("Get an Answer"):
    if question:
        response = multy_ai_agent.run(question)
        st.write(response)
