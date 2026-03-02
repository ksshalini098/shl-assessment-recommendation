from fastapi import FastAPI
from api.schemas import QueryRequest
from recommender.recommender import recommend_assessments

import os
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"message": "SHL Recommendation API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(request: QueryRequest):
    results = recommend_assessments(request.query)
    return {"recommendations": results}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)