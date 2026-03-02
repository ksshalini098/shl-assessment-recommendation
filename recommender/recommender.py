import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index("data/assessments.index")

print("Loading metadata...")
with open("data/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print("Recommender ready")


def recommend_assessments(query, top_k=10):

    # Convert query to embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS
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