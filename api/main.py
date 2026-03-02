from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API working"}

@app.get("/health")
def health():
    return {"status": "ok"}

# import recommender only after app starts
try:
    from api.schemas import QueryRequest
    from recommender.recommender import recommend_assessments

    @app.post("/recommend")
    def recommend(request: QueryRequest):
        results = recommend_assessments(request.query)
        return {"recommendations": results}

except Exception as e:
    print("Import error:", e)