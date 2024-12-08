import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./model")
tokenizer = DistilBertTokenizer.from_pretrained("./model")

# Predefined responses for categories
category_mapping = {
    0: "Billing Issue",
    1: "Cancellation Request",
    2: "Product Inquiry",
    3: "Refund Request",
    4: "Technical Issue",
}
responses = {
    0: "It seems like you’re facing a billing issue. Please contact our billing department.",
    1: "Your cancellation request has been received and processed. If there’s any feedback you’d like to share about your experience, we’d greatly appreciate it—it helps us improve!",
    2: "Let us know if you'd like to place an order or have further questions. We're here to help!",
    3: "Thank you for reaching out. We’ve received your request for a refund and are here to assist you. To process this, we’ll need specific information (your order number, proof of purchase). Once we have this, we’ll review your request and provide an update within 30 minutes.",
    4: "We’re actively looking into this issue and appreciate your patience. Our technical team is working on it, and we’ll keep you updated. In the meantime, if you have additional details, feel free to share.",
}

# Classify user query
def classify_ticket(query):
    debug_info = {"rule_based": False, "model_based": False, "rule_matched": None}

    # Rule-based hints for priority matching
    if any(word in query.lower() for word in ["charge", "charged", "billing", "refund"]):
        debug_info["rule_based"] = True
        debug_info["rule_matched"] = "Billing Issue"
        return 0, debug_info
    if any(word in query.lower() for word in ["cancel", "cancellation", "stop subscription", "terminate subscription"]):
        debug_info["rule_based"] = True
        debug_info["rule_matched"] = "Cancellation Request"
        return 1, debug_info
    if any(word in query.lower() for word in ["product", "specification", "order"]):
        debug_info["rule_based"] = True
        debug_info["rule_matched"] = "Product Inquiry"
        return 2, debug_info
    if any(word in query.lower() for word in ["refund", "return"]):
        debug_info["rule_based"] = True
        debug_info["rule_matched"] = "Refund Request"
        return 3, debug_info
    if any(word in query.lower() for word in ["login", "technical", "error", "issue"]):
        debug_info["rule_based"] = True
        debug_info["rule_matched"] = "Technical Issue"
        return 4, debug_info

    # Fallback to the model
    inputs = tokenizer.encode_plus(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(dim=1).item()

    # Ensure model output matches category_mapping keys
    if predicted_class not in category_mapping:
        debug_info["model_based"] = True
        debug_info["rule_matched"] = "Unknown"
        return "Unknown", debug_info

    debug_info["model_based"] = True
    return predicted_class, debug_info

# Streamlit GUI
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
        category_label, debug_info = classify_ticket(user_query)
        st.write(f"Debug Info: {debug_info}")  # Debugging output for troubleshooting
        if category_label == "Unknown":
            category = "Unknown Category"
            response = "Sorry, we couldn't classify your query. Please provide more details."
        else:
            category = category_mapping.get(category_label, "Unknown Category")
            response = responses.get(category_label, "We’re here to assist you. Please provide more details!")

    # Display Results
    st.success("Query Processed!")
    st.markdown(f"### *Category:* :blue[{category}]")
    st.markdown(f"### *Response:* :green[{response}]")
else:
    st.info("Enter a query to get started!")
