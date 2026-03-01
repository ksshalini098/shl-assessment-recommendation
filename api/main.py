from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --------------------
# App initialization
# --------------------
app = FastAPI(title="SHL Assessment Recommendation API")

# --------------------
# Load data & models (ONCE)
# --------------------
df = pd.read_csv("data/assessments_with_embeddings.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

index = faiss.read_index("data/assessments.index")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------
# Request schema
# --------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

# --------------------
# Health check endpoint
# --------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

# --------------------
# Recommendation endpoint
# --------------------
@app.post("/recommend")
def recommend_assessments(request: QueryRequest):
    query = request.query
    top_k = min(max(request.top_k, 1), 10)  # enforce 1–10

    # Embed query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # FAISS search
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