import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

df = pd.read_csv("dataset.csv")

X = df.drop("label", axis=1).values
y = df["label"]

# reshape into (samples, 21, 2)
X = X.reshape(-1, 21, 2)

# normalize relative to wrist
X_norm = []
for sample in X:
    base = sample[0]
    norm = sample - base
    X_norm.append(norm.flatten())

X_norm = np.array(X_norm)

model = KNeighborsClassifier(n_neighbors=1)  # faster + better

model.fit(X_norm, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained successfully")