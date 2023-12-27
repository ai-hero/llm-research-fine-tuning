"""Streamlit app for the AI Hero Fine Tune Research PoC."""
from typing import Generator

import streamlit as st
from huggingface_hub import InferenceClient

# Initialize the InferenceClient with the TGI server address
client = InferenceClient(model="http://127.0.0.1:8080")


def stream_response(prompt: str) -> Generator[str, str, str]:
    """Stream the response from the TGI server."""
    try:
        # Stream tokens from the response and yield them
        yield from client.text_generation(prompt, max_new_tokens=512, stream=True)
    except Exception as e:
        yield f"An error occurred: {e}"
    return ""


# Streamlit app interface
st.title("Fine Tune Research PoC")
st.subheader("Powered by AI Hero.")

# Text box for user input
user_input = st.text_area("Enter your prompt:", "What is Deep Learning?")
if not user_input.startswith("<s>"):
    user_input = "<s>" + user_input

st.text("Completion: ")
# Placeholder for the output
output_container = st.empty()

output_text = None
if output_text is None:
    output_container.info("Waiting for you to hit send...")

# Initialize the output text
output_text = ""

# Button to send the request
if st.button("Send"):
    # Stream the response from the TGI server and display output as it comes
    for part in stream_response(user_input):
        if part == "</s>":
            break
        # Append the new word to the output text
        output_text += part  # Add space between words
        # Update the output container with the new output text
        output_container.write(output_text)
