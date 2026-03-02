import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recommender.recommender import recommend_assessments


# Load test dataset
df = pd.read_excel("data/test_dataset.xlsx")

# Change column name if needed
query_column = df.columns[0]

queries = df[query_column].tolist()

output = []

for query in queries:

    results = recommend_assessments(query, top_k=10)

    for r in results:

        output.append({
            "Query": query,
            "Assessment_url": r["assessment_url"]
        })

# Save submission file
submission_df = pd.DataFrame(output)

submission_df.to_csv("evaluation/evaluate.csv", index=False)

print("Submission file generated successfully")