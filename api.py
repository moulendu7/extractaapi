import os, shutil, requests, redis, re
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

load_dotenv()

app = FastAPI()

UPLOAD_DIR = "/tmp/uploads"
FAISS_DIR = "/tmp/faiss"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")
REDIS_URL = os.getenv("REDIS_URL")

redis_client = redis.from_url(REDIS_URL)

class HFAPIEmbeddings(Embeddings):
    def __init__(self):
        self.api_url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction"
        self.headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    def embed_documents(self, texts):
        res = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": texts}
        )
        data = res.json()

        if isinstance(data, dict):
            raise Exception(f"HF Embedding API Error: {data}")

        if isinstance(data[0], float):
            data = [data]

        return data

    def embed_query(self, text):
        res = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": text}
        )
        data = res.json()

        if isinstance(data, dict):
            raise Exception(f"HF Embedding API Error: {data}")

        return data[0] if isinstance(data[0], list) else data
    def __init__(self):
        self.api_url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction"
        self.headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    def embed_documents(self, texts):
        res = requests.post(self.api_url, headers=self.headers, json={"inputs": texts})
        return res.json()

    def embed_query(self, text):
        res = requests.post(self.api_url, headers=self.headers, json={"inputs": text})
        return res.json()[0]

embeddings = HFAPIEmbeddings()

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/upload")
async def upload(file: UploadFile = File(...), user_id: str = ""):
    try:
        if not user_id:
            return {"message": "user_id required"}

        pdf_path = f"{UPLOAD_DIR}/{user_id}.pdf"

        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        docs = PyPDFLoader(pdf_path).load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=80
        )

        chunks = splitter.split_documents(docs)[:50]  # speed limit

        vs = FAISS.from_documents(chunks, embeddings)

        save_path = f"{FAISS_DIR}/{user_id}"
        os.makedirs(save_path, exist_ok=True)
        vs.save_local(save_path)

        redis_client.setex(f"user:{user_id}", 1800, "active")

        return {"message": "PDF stored"}

    except Exception as e:
        return {"error": str(e)}

def call_llm(context, question):
    url = "https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    prompt = f"""
Answer ONLY using the context.
If not found, say: Not found in document.

Context:
{context}

Question: {question}
"""

    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.3
    }

    res = requests.post(url, headers=headers, json=payload)
    data = res.json()

    return data["choices"][0]["message"]["content"]

@app.get("/ask")
def ask(user_id: str, question: str):
    try:
        if not redis_client.get(f"user:{user_id}"):
            return {"answer": "Session expired. Upload PDF again."}

        path = f"{FAISS_DIR}/{user_id}"

        vs = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs = vs.as_retriever(search_kwargs={"k": 4}).invoke(question)

        context = "\n\n".join(clean_text(d.page_content[:300]) for d in docs)

        answer = call_llm(context, question)

        return {"answer": answer}

    except Exception as e:
        return {"answer": str(e)}

@app.get("/reset")
def reset(user_id: str):
    redis_client.delete(f"user:{user_id}")
    return {"message": "reset"}

@app.get("/cleanup")
def cleanup(user_id: str):
    path = f"{FAISS_DIR}/{user_id}"

    if os.path.exists(path):
        shutil.rmtree(path)

    redis_client.delete(f"user:{user_id}")

    return {"message": "cleaned"}