import streamlit as st
from models.classifier import classify_ticket
from utils.helpers import chatbot_response

st.title("Resolvo")

user_query = st.text_input("Enter your support ticket query:")
if user_query:
    category = classify_ticket(user_query)
    response = chatbot_response(category, user_query)
    st.write(f"**Category:** {category}")
    st.write(f"**Response:** {response}")
