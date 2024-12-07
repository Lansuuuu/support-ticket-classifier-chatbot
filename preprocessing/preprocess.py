import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def clean_text(text):
    """Clean and normalize ticket text."""
    if pd.isna(text):  # Handle missing text
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def preprocess_data(filepath):
    data = pd.read_csv("/workspaces/support-ticket-classifier-chatbot/dataset/customer_support_tickets.csv")
    
    #Extract relevant columns
    data = data[['Ticket Description', 'Ticket Type']]
    
    #Remove duplicates and missing values
    data = data.drop_duplicates(subset=['Ticket Description'])
    data = data.dropna(subset=['Ticket Description', 'Ticket Type'])
    
    #Clean the text
    data['Ticket Description'] = data['Ticket Description'].apply(clean_text)
    
    #Encode ticket types as integers
    label_encoder = LabelEncoder()
    data['Ticket Type Label'] = label_encoder.fit_transform(data['Ticket Type'])
    
    #Save the category mapping for interpretation
    category_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print(f"Category Mapping: {category_mapping}")
    
    #Split into train, validation, and test sets
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    #Save the cleaned datasets
    train_data.to_csv("dataset/train_data.csv", index=False)
    val_data.to_csv("dataset/val_data.csv", index=False)
    test_data.to_csv("dataset/test_data.csv", index=False)
    print("Datasets saved successfully!")

preprocess_data("dataset/support_tickets.csv")