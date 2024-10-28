import streamlit as st
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import faiss

# 1. Load your dataset (Adapt this to your specific data)
def load_dataset(path):
    import os
    docs = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r") as f:
                docs.append(f.read())
    return docs

# 2. Initialize the RAG pipeline components
def initialize_rag_pipeline():
    # 2.1 Load LLM and Tokenizer
    model_name = "distilbert-base-cased-distilled-squad"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2.2 Load Sentence Transformer
    embedder = SentenceTransformer('all-mpnet-base-v2')

    # 2.3 Create a FAISS index
    index = faiss.IndexFlatL2(768)  
    return model, tokenizer, embedder, index

# 3. Create embeddings and populate FAISS index
def create_vector_store(docs, embedder, index):
    embeddings = embedder.encode(docs)
    index.add(embeddings)
    return index

# 4. Answer the user's question
def answer_question(question, docs, model, tokenizer, embedder, index):
    # 4.1 Embed the question
    query_embedding = embedder.encode(question)

    # 4.2 Search the vector store for similar documents
    k = 5  # Retrieve the top 5 most similar documents
    distances, indices = index.search(query_embedding, k)

    # 4.3 Retrieve the text of similar documents
    retrieved_texts = [docs[i] for i in indices[0]]

    # 4.4 Answer the question (using the LLM with retrieved context)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    answer = qa_pipeline(question=question, context=" ".join(retrieved_texts))
    return answer['answer']

# --- Streamlit Interface ---

st.title("RAG Chatbot (Streamlit)")

dataset_path = st.text_input("Dataset Directory:", "path/to/your/dataset")  # Replace with your dataset directory

if dataset_path:
    docs = load_dataset(dataset_path)
    model, tokenizer, embedder, index = initialize_rag_pipeline()
    index = create_vector_store(docs, embedder, index)

    question = st.text_input("Ask a question:")
    if question:
        answer = answer_question(question, docs, model, tokenizer, embedder, index)
        st.write(f"**Answer:** {answer}")
