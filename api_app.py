from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import faiss
import os

app = FastAPI(title="SHL Assessment Recommendation API")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global objects
df = None
index = None
model = None


# -----------------------------
# STARTUP EVENT (CRITICAL FIX)
# -----------------------------
@app.on_event("startup")
async def load_resources():
    global df, index, model

    print("Starting resource loading...")

    from sentence_transformers import SentenceTransformer

    # Load CSV
    df = pd.read_csv("data/assessments_with_embeddings.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Load FAISS index
    index = faiss.read_index("data/assessments.index")

    # ⚠️ Use smaller model for faster startup
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

    print("All resources loaded successfully!")


# -----------------------------
# REQUEST SCHEMA
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


# -----------------------------
# HEALTH ENDPOINT
# -----------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# -----------------------------
# RECOMMEND ENDPOINT
# -----------------------------
@app.post("/recommend")
async def recommend(request: QueryRequest):
    global df, index, model

    if model is None or index is None or df is None:
        return {"error": "Model not loaded yet. Please wait."}

    query = request.query
    top_k = min(max(request.top_k, 1), 10)

    # Encode query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append({
            "name": df.iloc[idx]["name"],
            "url": df.iloc[idx]["url"],
            "description": df.iloc[idx]["description"]
        })

    return {
        "query": query,
        "recommendations": results
    }


# -----------------------------
# MAIN (RENDER SAFE)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)