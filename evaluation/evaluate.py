import pandas as pd
from recommender.recommend import recommend_assessments

# Load train dataset
df = pd.read_excel("data/Gen_AI_Dataset.xlsx")

def recall_at_k(predicted, actual, k=10):
    predicted_k = predicted[:k]
    hits = len(set(predicted_k) & set(actual))
    return hits / len(actual) if len(actual) > 0 else 0

recalls = []

for _, row in df.iterrows():
    query = row["query"]

    # actual URLs (comma-separated)
    actual_urls = [url.strip() for url in row["assessment_url"].split(",")]

    # model predictions
    results = recommend_assessments(query, top_k=20)
    predicted_urls = [r["assessment_url"] for r in results]

    r_at_10 = recall_at_k(predicted_urls, actual_urls, 10)
    recalls.append(r_at_10)

mean_recall_10 = sum(recalls) / len(recalls)

print("✅ Mean Recall@10:", round(mean_recall_10, 4))
