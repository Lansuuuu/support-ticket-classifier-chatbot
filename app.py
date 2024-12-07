import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

#Load trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./model")
tokenizer = DistilBertTokenizer.from_pretrained("./model")

#Predefined responses for categories
category_mapping = {
    0: "Billing Issue",
    1: "Technical Support",
    2: "Account Issue",
}
responses = {
    0: "It seems like you’re facing a billing issue. Please contact our billing department.",
    1: "This appears to be a technical issue. Restarting your device might help.",
    2: "Account issues require password resets. Please follow the instructions in your email.",
}

#Classify user query
def classify_ticket(query):
    inputs = tokenizer.encode_plus(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(dim=1).item()
    return predicted_class

#Streamlit GUI
st.title("Resolvo")

user_query = st.text_input("Enter your support ticket query:")
if user_query:
    category_label = classify_ticket(user_query)
    category = category_mapping.get(category_label, "Unknown Category")
    response = responses.get(category_label, "We’re here to assist you. Please provide more details!")
    st.write(f"**Category:** {category}")
    st.write(f"**Response:** {response}")
