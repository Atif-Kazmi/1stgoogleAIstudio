import streamlit as st
from transformers import pipeline

st.title("PDF Chatbot (Open-Source)")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    # Load the PDF and extract text (using PyPDF2)
    import PyPDF2
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Load a pre-trained question-answering model
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    # Chat interface
    question = st.text_input("Ask your question:")

    if question:
        # Get the answer
        answer = qa_pipeline(question=question, context=text)
        st.write(f"**Answer:** {answer['answer']}")
