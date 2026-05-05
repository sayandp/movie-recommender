import pandas as pd
import pickle
import os
import json
import time
import argparse
from sklearn.neighbors import NearestNeighbors
from app.data_loader import load_ratings

MODEL_PATH    = "model/knn_model.pkl"
MATRIX_PATH   = "model/user_item_matrix.pkl"
METADATA_PATH = "model/knn_metadata.json"


def train_model(k: int = 20, metric: str = "cosine") -> dict:
    print("Loading ratings...")
    df = load_ratings()
    print(f"  {len(df):,} ratings | {df['user_id'].nunique():,} users | {df['item_id'].nunique():,} items")

    print("Building user-item matrix...")
    matrix = df.pivot(
        index="user_id",
        columns="item_id",
        values="rating",
    ).fillna(0)
    print(f"  Matrix shape: {matrix.shape}  (sparsity: {(matrix == 0).values.mean():.1%})")

    print(f"\nTraining KNN (k={k}, metric={metric})...")
    t0    = time.time()
    model = NearestNeighbors(
        n_neighbors=k + 1,
        metric=metric,
        algorithm="brute",
        n_jobs=-1,
    )
    model.fit(matrix.values)
    elapsed = round(time.time() - t0, 2)
    print(f"  Done in {elapsed}s")

    os.makedirs("model", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(MATRIX_PATH, "wb") as f:
        pickle.dump(matrix, f)

    metadata = {
        "k":            k,
        "metric":       metric,
        "n_users":      int(df["user_id"].nunique()),
        "n_items":      int(df["item_id"].nunique()),
        "n_ratings":    len(df),
        "matrix_shape": list(matrix.shape),
        "sparsity":     round(float((matrix == 0).values.mean()), 4),
        "train_time_s": elapsed,
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel    → {MODEL_PATH}")
    print(f"Matrix   → {MATRIX_PATH}")
    print(f"Metadata → {METADATA_PATH}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",      type=int, default=20,       help="Number of neighbours")
    parser.add_argument("--metric", type=str, default="cosine", help="cosine | euclidean | manhattan")
    args = parser.parse_args()
    train_model(k=args.k, metric=args.metric)