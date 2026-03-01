import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("🚀 Embedding script started")

# Load assessment data
df = pd.read_csv("data/assessments.csv")

print("✅ CSV loaded")
print("Number of assessments:", len(df))
print("Columns:", df.columns)

# Use only assessment names
texts = (
    df["assessment_name"].fillna("") + " " +
    df["assessment_description"].fillna("")
).str.lower().tolist()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded")

# Generate embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# Convert to float32
embeddings = np.array(embeddings).astype("float32")

print("✅ Embeddings generated")
print("Embedding shape:", embeddings.shape)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to index
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "data/assessments.index")

# Save metadata
df.to_csv("data/assessments_with_embeddings.csv", index=False)

print("🎉 STEP 4 COMPLETED SUCCESSFULLY")