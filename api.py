from fastapi import FastAPI, UploadFile, File
import os, shutil, re, requests, pickle, redis

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings


app = FastAPI()

FAISS_DIR = "/tmp/faiss"
os.makedirs(FAISS_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"status": "API running"}

redis_url = os.getenv("REDIS_URL")
redis_client = redis.from_url(redis_url)

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5",
    huggingfacehub_api_token=HF_TOKEN
)

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

@app.post("/upload")
async def upload(file: UploadFile = File(...), user_id: str = ""):
    try:
        path = f"/tmp/{user_id}.pdf"

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        docs = PyPDFLoader(path).load()

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100
        ).split_documents(docs)

        vs = FAISS.from_documents(chunks, embeddings)

        save_path = f"{FAISS_DIR}/{user_id}"
        os.makedirs(save_path, exist_ok=True)

        vs.save_local(save_path)

        redis_client.setex(f"user:{user_id}", 1800, "active")

        return {"message": "PDF stored"}

    except Exception as e:
        return {"message": str(e)}

def call_llm(context, question):
    try:
        url = "https://router.huggingface.co/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }

        prompt = f"""
You are a helpful AI assistant.

Answer using the context.
If not found, say Not found in document.

Context:
{context}

Question: {question}

Answer:
"""

        payload = {
            "model": "HuggingFaceH4/zephyr-7b-beta",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300
}

        res = requests.post(url, headers=headers, json=payload)

        if res.status_code != 200:
            return f"LLM error: {res.text}"

        data = res.json()

        return data.get("choices", [{}])[0].get("message", {}).get("content", "No answer")

    except Exception as e:
        return f"LLM error: {str(e)}"

@app.get("/ask")
def ask(user_id: str, question: str):
    try:
        if not redis_client.get(f"user:{user_id}"):
            return {"answer": "Upload PDF first"}

        vs = FAISS.load_local(
            f"{FAISS_DIR}/{user_id}",
            embeddings,
            allow_dangerous_deserialization=True
        )
        docs = vs.as_retriever(search_kwargs={"k": 4}).invoke(question)
        context = "\n\n".join(d.page_content[:400] for d in docs)
        answer = call_llm(context, question)
        return {"answer": answer}
    except Exception as e:
        return {"answer": str(e)}

@app.get("/reset")
def reset(user_id: str):
    redis_client.delete(f"user:{user_id}")
    return {"message": "reset done"}

import shutil

@app.get("/cleanup")
def cleanup(user_id: str):
    path = f"{FAISS_DIR}/{user_id}"
    if os.path.exists(path):
        shutil.rmtree(path)
    redis_client.delete(f"user:{user_id}")
    return {"message": "cleaned"}