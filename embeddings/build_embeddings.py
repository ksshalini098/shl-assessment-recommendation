import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

print("🚀 Embedding script started")

df = pd.read_csv("data/assessments.csv")

print("✅ CSV loaded:", len(df))

texts = (
    df["name"].fillna("") + " " +
    df["description"].fillna("")
).str.lower().tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")

print("✅ Model loaded")

embeddings = model.encode(texts, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(index, "data/assessments.index")

metadata = df.to_dict("records")

with open("data/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("🎉 Embeddings and metadata saved successfully")