import streamlit as st
from app import predict_intent

st.set_page_config(page_title="Automotive AI", page_icon="🚗")

st.title("🚗 Automotive Voice Assistant")
st.write("AI-powered intent classification for in-car voice commands")

user_input = st.text_input("Enter your command:")

if st.button("Predict Intent"):
    if user_input:
        result = predict_intent(user_input)

        st.success(f"Intent: {result['Predicted Intent']}")
        st.info(f"Confidence: {result['Confidence']}%")
    else:
        st.warning("Please enter a command")
