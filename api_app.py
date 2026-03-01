from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="SHL Assessment Recommendation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (OK for assignment)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Global variables
df = None
index = None
model = None


# --------------------
# Load everything at startup (VERY IMPORTANT)
# --------------------
@app.on_event("startup")
def load_models():
    global df, index, model

    print("Loading CSV...")
    df = pd.read_csv("data/assessments_with_embeddings.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    print("Loading FAISS index...")
    index = faiss.read_index("data/assessments.index")

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("All resources loaded successfully!")


# --------------------
# Request schema
# --------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


# --------------------
# Health check
# --------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# --------------------
# Recommendation endpoint
# --------------------
@app.post("/recommend")
def recommend_assessments(request: QueryRequest):
    global df, index, model

    query = request.query
    top_k = min(max(request.top_k, 1), 10)

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append({
            "assessment_name": df.iloc[idx]["assessment_name"],
            "assessment_url": df.iloc[idx]["assessment_url"]
        })

    return {
        "query": query,
        "recommendations": results
    }