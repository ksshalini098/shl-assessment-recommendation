import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load assessment metadata
df = pd.read_csv("data/assessments_with_embeddings.csv")

# Load FAISS index
index = faiss.read_index("data/assessments.index")

# Load same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def recommend_assessments(query, top_k=10):
    """
    Input: job description text
    Output: list of recommended assessments (name + url)
    """

    # Convert query to embedding
    query = query.lower().strip()
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    query = query.lower().strip()

     # Boost technical terms slightly
    if "java" in query:
     query += " java programming backend development"
    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)

    recommendations = []

    for idx in indices[0]:
        recommendations.append({
            "assessment_name": df.iloc[idx]["assessment_name"],
            "assessment_url": df.iloc[idx]["assessment_url"]
        })

    return recommendations


# ---- Test run ----
if __name__ == "__main__":
    test_query = "Hiring a Java developer with good communication and teamwork skills"
    results = recommend_assessments(test_query)

    for r in results:
        print(r["assessment_name"], "->", r["assessment_url"])