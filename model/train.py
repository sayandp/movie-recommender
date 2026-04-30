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
    """
    Train user-based KNN collaborative filter.

    Steps
    -----
    1. Load MovieLens 100K ratings
    2. Build user-item matrix (pivot, fill 0 for unseen)
    3. Fit NearestNeighbors on the matrix
    4. Save model + matrix + metadata
    """

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("Loading ratings...")
    df = load_ratings()
    print(f"  {len(df):,} ratings | {df['user_id'].nunique():,} users | {df['item_id'].nunique():,} items")

    # ── 2. Build matrix ───────────────────────────────────────────────────────
    print("Building user-item matrix...")
    matrix = df.pivot(
        index="user_id",
        columns="item_id",
        values="rating",
    ).fillna(0)
    print(f"  Matrix shape: {matrix.shape}  (sparsity: {(matrix == 0).values.mean():.1%})")

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print(f"\nTraining KNN (k={k}, metric={metric})...")
    t0 = time.time()
    model = NearestNeighbors(
        n_neighbors=k + 1,   # +1 so we can exclude the query user itself
        metric=metric,
        algorithm="brute",   # best for sparse high-dimensional data
        n_jobs=-1,
    )
    model.fit(matrix.values)
    elapsed = round(time.time() - t0, 2)
    print(f"  Done in {elapsed}s")

    # ── 4. Save ───────────────────────────────────────────────────────────────
    os.makedirs("model", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(MATRIX_PATH, "wb") as f:
        pickle.dump(matrix, f)

    metadata = {
        "k":             k,
        "metric":        metric,
        "n_users":       int(df["user_id"].nunique()),
        "n_items":       int(df["item_id"].nunique()),
        "n_ratings":     len(df),
        "matrix_shape":  list(matrix.shape),
        "sparsity":      round(float((matrix == 0).values.mean()), 4),
        "train_time_s":  elapsed,
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved:")
    print(f"  Model    → {MODEL_PATH}")
    print(f"  Matrix   → {MATRIX_PATH}")
    print(f"  Metadata → {METADATA_PATH}")

    print("\n── Summary ─────────────────────────────────────")
    for key, val in metadata.items():
        print(f"  {key:<20} {val}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KNN Movie Recommender")
    parser.add_argument("--k",      type=int, default=20,       help="Number of neighbours (default: 20)")
    parser.add_argument("--metric", type=str, default="cosine", help="Distance metric: cosine | euclidean | manhattan")
    args = parser.parse_args()

    train_model(k=args.k, metric=args.metric)
    train_model()