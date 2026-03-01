from recommender.recommend import recommend_assessments

query = """
I am hiring for Java developers who can also collaborate effectively
with my business teams. Looking for an assessment that can be completed
in 40 minutes.
"""

print("Query:")
print(query)
print("\nRecommended Assessments:\n")

results = recommend_assessments(query, top_k=5)

for i, r in enumerate(results, 1):
    print(f"{i}. {r}")