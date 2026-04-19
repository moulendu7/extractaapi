from fastapi import FastAPI, UploadFile, File
import os, shutil, re, requests, redis
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

app = FastAPI()

UPLOAD_DIR = "/tmp/uploads"
FAISS_DIR = "/tmp/faiss"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")
REDIS_URL = os.getenv("REDIS_URL")

if not HF_TOKEN:
    raise Exception("HF_TOKEN missing")

if not REDIS_URL:
    raise Exception("REDIS_URL missing")

redis_client = redis.from_url(REDIS_URL)

embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5",
    huggingfacehub_api_token=HF_TOKEN
)

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()
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

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100
        ).split_documents(docs)

        vs = FAISS.from_documents(chunks, embeddings)

        save_path = f"{FAISS_DIR}/{user_id}"
        os.makedirs(save_path, exist_ok=True)
        vs.save_local(save_path)

        redis_client.setex(f"user:{user_id}", 1800, "active")

        return {"message": "PDF stored successfully"}

    except Exception as e:
        return {"error": str(e)}

def call_llm(context, question):
    try:
        url = "https://router.huggingface.co/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }

        prompt = f"""
Answer ONLY using the context below.
If answer not found, say: Not found in document.

Context:
{context}

Question: {question}
"""

        payload = {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300,
            "temperature": 0.3
        }

        res = requests.post(url, headers=headers, json=payload)

        if res.status_code != 200:
            return f"LLM error: {res.text}"

        data = res.json()

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"LLM error: {str(e)}"
@app.get("/ask")
def ask(user_id: str, question: str):
    try:
        if not user_id:
            return {"answer": "user_id required"}

        if not redis_client.get(f"user:{user_id}"):
            return {"answer": "Session expired. Upload PDF again."}

        path = f"{FAISS_DIR}/{user_id}"

        if not os.path.exists(path):
            return {"answer": "Vector DB not found. Upload PDF again."}

        vs = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs = vs.as_retriever(search_kwargs={"k": 4}).invoke(question)

        context = "\n\n".join(clean_text(d.page_content[:400]) for d in docs)

        answer = call_llm(context, question)

        return {"answer": answer}

    except Exception as e:
        return {"answer": str(e)}

@app.get("/reset")
def reset(user_id: str):
    redis_client.delete(f"user:{user_id}")
    return {"message": "Session reset"}

@app.get("/cleanup")
def cleanup(user_id: str):
    path = f"{FAISS_DIR}/{user_id}"

    if os.path.exists(path):
        shutil.rmtree(path)

    redis_client.delete(f"user:{user_id}")

    return {"message": "Cleaned successfully"}