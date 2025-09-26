from utils import chunk_text
from embedding_utils import get_text_embeddings
from llm_response import generate_answer_from_prompt
import streamlit as st
import psycopg2

# -------------------- Database Connection --------------------
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="postgres",
    user="postgres",
    password="postgres"
)
cursor = conn.cursor()

# Ensure pgvector extension and table exist
cursor.execute("""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(768)
);
""")
conn.commit()

# -------------------- Static Documents --------------------
documents = [
    "AI is a rapidly advancing field focused on creating intelligent machines that can simulate human cognitive functions like learning, problem-solving, and decision-making.",
    "From powering recommendation engines and self-driving cars to assisting in medical diagnostics, AI has become an integral part of modern technology, driving innovation across various industries.",
    "The future of AI holds immense potential to revolutionize how we live and work, while also raising important ethical considerations about data privacy, bias, and the societal impact of automation."
]

# -------------------- Streamlit App --------------------
st.set_page_config(page_title="RAG QnA Pipeline", layout="wide")

st.title("📚 Interactive QnA System with Retrieval-Augmented Generation (RAG)")
st.markdown("""
Welcome! This application demonstrates a simple **RAG pipeline**:
1. Chunk documents into smaller pieces.
2. Generate embeddings for each chunk.
3. Store embeddings in PostgreSQL with **pgvector**.
4. Retrieve the most relevant chunks for a user query.
5. Generate a final answer using a language model.
""")

# -------------------- Step 1: Document Chunking --------------------
st.header("Step 1: Document Chunking")
st.markdown("Split each document into smaller chunks to improve retrieval accuracy.")

chunks = chunk_text(documents, chunk_size=30)  # smaller chunks for demo
st.write(f"Number of chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks, start=1):
    st.write(f"**Chunk {i}:** {chunk}")

# -------------------- Step 2: Generate Embeddings --------------------
st.header("Step 2: Embedding Generation")
st.markdown("Generate vector embeddings for each chunk using Google AI's `text-embedding-004` model.")

embeddings = get_text_embeddings(chunks)
for chunk, emb in zip(chunks, embeddings):
    cursor.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        (chunk, emb)
    )
conn.commit()
st.success("Embeddings generated and stored in PostgreSQL!")

# -------------------- Step 3: Retrieve Relevant Chunks --------------------
st.header("Step 3: Retrieve Relevant Chunks")
st.markdown("Enter a question and the system will retrieve the top-k most relevant chunks using vector similarity.")

question = st.text_input("Enter your question:")

def get_relevant_chunks(question, top_n=3):
    question_embedding = get_text_embeddings([question])[0]
    cursor.execute("""
        SELECT content 
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (question_embedding, top_n))
    return [row[0] for row in cursor.fetchall()]

if question:
    relevant_chunks = get_relevant_chunks(question)
    st.write("### Retrieved Chunks:")
    for i, chunk in enumerate(relevant_chunks, start=1):
        st.write(f"{i}. {chunk}")

    # -------------------- Step 4: Generate Answer --------------------
    st.header("Step 4: Generate Answer Using Google AI Model")
    st.markdown("The retrieved chunks are used as context for generating a final answer to your question.")
    
    context = "\n".join(f"{i+1}. {chunk}" for i, chunk in enumerate(relevant_chunks))
    prompt = f"Using the following information:\n{context}\nAnswer the question: {question}"
    
    st.info("Generating response...")
    response = generate_answer_from_prompt(prompt)
    st.markdown("### ✅ Generated Answer:")
    st.write(response)

# -------------------- Cleanup --------------------
cursor.close()
conn.close()
