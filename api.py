from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import re
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
app = FastAPI()

@app.get("/")
def home():
    return {"status": "API running ✅"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
FAISS_DIR = "faiss"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN,
    model_name="BAAI/bge-small-en-v1.5"
)

# -------- CLEAN --------
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()


@app.post("/upload")
async def upload(file: UploadFile = File(...), user_id: str = ""):
    file_path = f"{UPLOAD_DIR}/{user_id}.pdf"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(f"{FAISS_DIR}/{user_id}")

    return {"message": "PDF processed successfully"}


def call_llm(context, question):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    prompt = f"""
Answer clearly and in detail using ONLY the context below.

Context:
{context}

Question: {question}

Answer:
"""

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": prompt}
    )

    result = response.json()

    try:
        return result[0]["generated_text"]
    except:
        return "⚠️ LLM error. Try again."


@app.get("/ask")
def ask(user_id: str, question: str):
    try:
        vectorstore = FAISS.load_local(f"{FAISS_DIR}/{user_id}", embeddings)
    except:
        return {"answer": "❌ Upload PDF first."}

    docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(question)

    context = "\n\n".join([
        clean_text(d.page_content)[:400] for d in docs
    ])

    answer = call_llm(context, question)

    return {"answer": answer}