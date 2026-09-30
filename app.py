import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from google import genai

st.title("PDF Chatbot")

key = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=key)
model = SentenceTransformer("all-MiniLM-L6-v2")

pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:
    reader = PdfReader(pdf)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    words = text.split()
    chunks = []

    for i in range(0, len(words), 150):
        chunks.append(" ".join(words[i:i + 150]))

    vectors = model.encode(chunks)

    question = st.text_input("Ask a question")

    if question:
        q = model.encode([question])[0]
        scores = np.dot(vectors, q)
        ids = np.argsort(scores)[-3:]

        context = ""

        for i in ids:
            context += chunks[i] + "\n"

        prompt = f"""
Answer using only the PDF text below.

If the answer is not in the PDF, say:
"I couldn't find that information in the uploaded PDF."

PDF:
{context}

Question:
{question}
"""

try:
    result = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )
    st.write(result.text)

except Exception:
    st.write("I couldn't find that information in the uploaded PDF.")
