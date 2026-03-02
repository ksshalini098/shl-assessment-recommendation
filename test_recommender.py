from recommender.recommender import recommend_assessments

query = "sales graduate"

results = recommend_assessments(query)

for r in results:
    print(r["assessment_name"])
    print(r["assessment_url"])
    print()