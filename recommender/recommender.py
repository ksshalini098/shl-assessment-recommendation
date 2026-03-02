import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load index and metadata immediately (fast)
index_path = os.path.join(BASE_DIR, "data", "assessments.index")
metadata_path = os.path.join(BASE_DIR, "data", "metadata.pkl")

index = faiss.read_index(index_path)

with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

# Model will load only when needed
model = None


def get_model():
    global model
    if model is None:
        print("Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def recommend_assessments(query, top_k=10):

    model_instance = get_model()

    query_embedding = model_instance.encode([query])
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