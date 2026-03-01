# SHL Assessment Recommendation System (Generative AI Assignment)

##  Overview

This project is a semantic recommendation system that suggests relevant SHL assessments based on natural language hiring requirements.

Instead of traditional keyword matching, the system uses transformer-based embeddings and vector similarity search to understand the meaning of the hiring query and return the most relevant assessments.

---

##  Features

- Accepts natural language hiring queries
- Uses semantic similarity (not keyword search)
- Fast vector search using FAISS
- REST API built with FastAPI
- Simple frontend for live demo
- Returns top relevant SHL assessments with URLs

---

##  System Architecture

1. Assessment descriptions are converted into vector embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).
2. Embeddings are stored and indexed using FAISS for efficient similarity search.
3. User hiring query is converted into an embedding.
4. Cosine similarity search is performed against stored assessment embeddings.
5. Top matching assessments are returned through a REST API.
6. Frontend displays recommendations in a simple UI.

---

##  Tech Stack

- Python
- FastAPI
- Sentence Transformers
- FAISS (Vector Similarity Search)
- Pandas & NumPy
- HTML / JavaScript (Frontend)

---

##  Project Structure

SHL_RECOMMENDATION/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   ├── assessments.csv
│   └── assessments_with_embeddings.csv
│
├── embeddings/
│   ├── __init__.py
│   └── build_embeddings.py
│
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py
│
├── frontend/
│   └── index.html
│
├── recommender/
│   ├── __init__.py
│   └── recommend.py
│
├── api_app.py
├── test_manual.py
└── README.md

## ⚙️ Installation & Setup

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Backend API
uvicorn api_app:app --reload

Backend will start at:

http://127.0.0.1:8000

4️⃣ Open Frontend

Open:

frontend/index.html

in your browser.

🧪 Example Query
I am hiring for Java developers who can collaborate with business teams. 
Need an assessment completed in 40 minutes.

✅ Output

Returns top matching SHL assessments
Includes assessment name
Includes clickable SHL product URL

📌 Design Decisions

Transformer embeddings used for semantic understanding.
FAISS used for efficient vector similarity search.
FastAPI used for lightweight REST API creation.
Simple frontend created for demonstration purposes.

📊 Why Semantic Search?

Traditional keyword search fails when:
Different wording is used
Synonyms are present
Context matters
Semantic search captures meaning and intent rather than exact word matches.

🛠️ Future Improvements

Add filtering by duration (e.g., 30–60 mins)
Add skill tagging metadata
Improve ranking using hybrid search (keyword + semantic)
Deploy as a cloud-hosted API

🎯 Assignment Objective

The goal of this project is to demonstrate the ability to:
Work with transformer-based embeddings
Implement semantic search
Build an API-based solution
Deliver an end-to-end working system