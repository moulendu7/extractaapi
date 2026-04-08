from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, re, requests, pickle, redis
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

app = FastAPI()

@app.get("/")
def home():
    return {"status": "API running ✅"}

redis_client = redis.from_url(os.getenv("REDIS_URL"))

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN,
    model_name="BAAI/bge-small-en-v1.5"
)

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def save_vectorstore(user_id, vs):
    redis_client.setex(
        f"user:{user_id}",
        1800,  # 30 min expiry
        pickle.dumps(vs)
    )

def load_vectorstore(user_id):
    data = redis_client.get(f"user:{user_id}")
    if not data:
        return None
    return pickle.loads(data)

@app.post("/upload")
async def upload(file: UploadFile = File(...), user_id: str = ""):
    path = f"{UPLOAD_DIR}/{user_id}.pdf"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    docs = PyPDFLoader(path).load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    ).split_documents(docs)

    vs = FAISS.from_documents(chunks, embeddings)

    save_vectorstore(user_id, vs)

    return {"message": "PDF stored (30 min expiry)"}

def call_llm(context, question):
    url = "https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are a helpful AI assistant.

Answer clearly using the context below.
If answer is not present, say: Not found in document.

Context:
{context}

Question: {question}

Answer:
"""

    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300
    }

    res = requests.post(url, headers=headers, json=payload)

    try:
        return res.json()["choices"][0]["message"]["content"]
    except:
        return f"⚠️ LLM error: {res.text}"

@app.get("/ask")
def ask(user_id: str, question: str):
    vs = load_vectorstore(user_id)

    if not vs:
        return {"answer": "❌ Upload PDF first (or expired after 30 min)."}

    docs = vs.as_retriever(search_kwargs={"k": 4}).invoke(question)

    context = "\n\n".join(
        clean_text(d.page_content)[:500] for d in docs
    )

    return {"answer": call_llm(context, question)}

@app.get("/reset")
def reset(user_id: str):
    redis_client.delete(f"user:{user_id}")
    return {"message": "Reset done"}