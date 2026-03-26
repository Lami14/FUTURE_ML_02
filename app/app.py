import streamlit as st
from src.predict import predict_ticket

# Page config
st.set_page_config(page_title="Support Ticket Classifier", layout="centered")

# Title
st.title("🎫 Support Ticket Classification System")

st.write("Enter a support ticket and let the ML model classify it and assign priority.")

# Input
user_input = st.text_area("✍️ Enter Support Ticket:", height=150)

# Button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a ticket.")
    else:
        result = predict_ticket(user_input)

        st.success("Prediction Complete ✅")

        # Output
        st.subheader("📊 Results")
        st.write(f"**Category:** {result['category']}")
        st.write(f"**Priority:** {result['priority']}")
