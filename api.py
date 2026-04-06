from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)


tokenizer = None
model = None

def load_model():
    global tokenizer, model

    if tokenizer is None:
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        print("Model loaded ✅")

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()


@app.post("/upload")
async def upload(file: UploadFile = File(...), user_id: str = ""):
    file_path = f"{UPLOAD_DIR}/{user_id}.pdf"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(f"{FAISS_DIR}/{user_id}")

    return {"message": "PDF processed successfully"}

@app.get("/ask")
def ask(user_id: str, question: str):

    load_model()  # 🔥 KEY FIX

    try:
        vectorstore = FAISS.load_local(f"{FAISS_DIR}/{user_id}", embeddings)
    except:
        return {"answer": "❌ Upload PDF first."}

    docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(question)

    context = "\n\n".join([clean_text(d.page_content)[:400] for d in docs])

    prompt = f"""
Answer clearly and in detail using ONLY the context.

Context:
{context}

Question: {question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"answer": answer}