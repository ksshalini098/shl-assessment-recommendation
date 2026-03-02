import pandas as pd

df = pd.read_csv("data/assessments.csv")

# keywords to remove
unwanted = [
    "guide",
    "framework",
    "profiling",
    "interview",
    "pre-packaged",
    "job focus",
    "competency"
]

# filter rows
mask = ~df["name"].str.lower().str.contains("|".join(unwanted), na=False)

clean_df = df[mask]

clean_df.to_csv("data/assessments_cleaned.csv", index=False)

print("Before cleaning:", len(df))
print("After cleaning:", len(clean_df))