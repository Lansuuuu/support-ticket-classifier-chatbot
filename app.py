import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

#Load trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./model")
tokenizer = DistilBertTokenizer.from_pretrained("./model")

#Predefined responses for categories
category_mapping = {
    0: "Billing Inquiry",
    1: "Cancellation Request",
    2: "Product Inquiry",
    3: "Refund request",
    4: "Technical Issue",
}
responses = {
    0: "It seems like you’re facing a billing issue. Please contact our billing department.",
    1: "Your cancellation request has been received and processed. If there’s any feedback you’d like to share about your experience, we’d greatly appreciate it—it helps us improve!",
    2: "Let us know if you'd like to place an order or have further questions. We're here to help!",
    3: "Thank you for reaching out. We’ve received your request for a refund and are here to assist you. To process this, we’ll need specific information (your order number, proof of purchase). Once we have this, we’ll review your request and provide an update within 30 minutes.",
    4: "We’re actively looking into this issue and appreciate your patience. Our technical team is working on it, and we’ll keep you updated. In the meantime, if you have additional details, feel free to share.",
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
st.set_page_config(
    page_title="Resolvo - AI Support Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.image("https://via.placeholder.com/300x100.png?text=Resolvo", use_container_width=True)
st.sidebar.title("About Resolvo")
st.sidebar.write(
    """
    Resolvo 
    - *Enter your query* in the input box.
    - *Receive instant assistance* based on your category.
    """
)

# Header
st.title("🤖 Welcome to Resolvo!")
st.markdown(
    """
    Resolvo helps classify your support queries and provides immediate assistance.
    Start by entering your issue below!
    """
)

# User Input
st.subheader("Enter Your Support Ticket Query")
user_query = st.text_input("What's your concern today?", "")

# Classification and Response
if user_query:
    with st.spinner("Processing your query..."):
        category_label = classify_ticket(user_query)
        category = category_mapping.get(category_label, "Unknown Category")
        response = responses.get(category_label, "We’re here to assist you. Please provide more details!")

    # Display Results
    st.success("Query Processed!")
    st.markdown(f"### *Category:* :blue[{category}]")
    st.markdown(f"### *Response:* :green[{response}]")
else:
    st.info("Enter a query to get started!")
