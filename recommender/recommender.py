import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load assessment metadata
df = pd.read_csv("data/assessments_with_embeddings.csv")
df.columns = df.columns.str.strip().str.lower()

# Load FAISS index
index = faiss.read_index("data/assessments.index")

# Load embedding model
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

def recommend_assessments(query, top_k=10):
    """
    Input: job description text
    Output: list of recommended assessments
    """

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    recommendations = []

    for idx in indices[0]:
        recommendations.append({
            "name": df.iloc[idx]["name"],
            "url": df.iloc[idx]["url"],
            "description": df.iloc[idx]["description"]
        })

    return recommendations


# ---- Test run ----
if __name__ == "__main__":
    test_query = "Hiring a Java developer with strong communication and teamwork skills"
    results = recommend_assessments(test_query)

    for r in results:
        print(r["name"], "->", r["url"])