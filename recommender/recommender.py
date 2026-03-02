import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index_path = os.path.join(BASE_DIR, "data", "assessments.index")
index = faiss.read_index(index_path)

print("Loading metadata...")
metadata_path = os.path.join(BASE_DIR, "data", "metadata.pkl")
with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

print("Recommender ready")


def recommend_assessments(query, top_k=10):

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx in indices[0]:
        item = metadata[idx]
        results.append({
            "assessment_name": item["name"],
            "assessment_url": item["url"],
            "description": item["description"]
        })

    return results