from fastapi import FastAPI
from pydantic import BaseModel
from recommender.recommender import recommend_assessments

app = FastAPI(title="SHL Assessment Recommendation API")

# Request model
class QueryRequest(BaseModel):
    query: str


# Health endpoint (MANDATORY for SHL)
@app.get("/health")
def health():
    return {"status": "ok"}


# Recommendation endpoint (MANDATORY for SHL)
@app.post("/recommend")
def recommend(request: QueryRequest):

    query = request.query

    results = recommend_assessments(query, top_k=10)

    return {
        "query": query,
        "recommendations": results
    }