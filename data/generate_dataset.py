import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=5,
                           n_informative=5, n_redundant=0, n_clusters_per_class=4, scale=10, random_state=42)

df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4", "x5"])
df["label"] = y

print(df.head())
print(f"Label distribution:\n{df['label'].value_counts()}")

df.to_csv("data.csv", index=False)
