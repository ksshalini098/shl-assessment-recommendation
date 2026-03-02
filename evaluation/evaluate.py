import pandas as pd
import sys
import os

# -------------------------------------------------
# Add project root to Python path
# -------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from recommender.recommender import recommend_assessments

# -------------------------------------------------
# Load SHL test dataset
# -------------------------------------------------
df = pd.read_excel("data/Gen_AI_Dataset.xlsx")

def recall_at_k(predicted, actual, k=10):
    predicted_k = predicted[:k]
    hits = len(set(predicted_k) & set(actual))
    return hits / len(actual) if len(actual) > 0 else 0

recalls = []

for _, row in df.iterrows():
    query = row["query"]

    # Actual URLs from SHL dataset
    actual_urls = [u.strip() for u in row["assessment_url"].split(",")]

    # Model predictions
    results = recommend_assessments(query, top_k=10)
    predicted_urls = [r["url"] for r in results]

    r_at_10 = recall_at_k(predicted_urls, actual_urls, 10)
    recalls.append(r_at_10)

mean_recall_10 = sum(recalls) / len(recalls)
print("✅ Mean Recall@10:", round(mean_recall_10, 4))