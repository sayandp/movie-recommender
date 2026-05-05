"""
evaluate.py — Proper hold-out evaluation for KNN Movie Recommender

Uses train/test split:
- 80% of each user's ratings = training
- 20% of each user's ratings = test (held out, treated as unseen)
- We check if model recommends the held-out liked movies

Run:
    python evaluate.py
    python evaluate.py --k 10 --users 100 --threshold 4.0
    python evaluate.py --compare-k
"""

import argparse
import time
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.data_loader import load_ratings, load_movies
from sklearn.neighbors import NearestNeighbors


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
ratings_df   = load_ratings()
movies_df    = load_movies()
movie_titles = dict(zip(movies_df["item_id"], movies_df["title"]))
all_items    = sorted(ratings_df["item_id"].unique())


# ── Train/test split ──────────────────────────────────────────────────────────
def split_ratings(df, test_ratio=0.2, seed=42):
    """Hold out test_ratio of each user's ratings as test set."""
    np.random.seed(seed)
    train_rows, test_rows = [], []
    for _, group in df.groupby("user_id"):
        group  = group.sample(frac=1, random_state=seed)
        n_test = max(1, int(len(group) * test_ratio))
        test_rows.append(group.iloc[:n_test])
        train_rows.append(group.iloc[n_test:])
    return pd.concat(train_rows), pd.concat(test_rows)


print("Splitting 80% train / 20% test per user...")
train_df, test_df = split_ratings(ratings_df)
print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")


# ── Build train matrix ────────────────────────────────────────────────────────
print("Building train matrix...")
train_matrix = (
    train_df.pivot(index="user_id", columns="item_id", values="rating")
    .fillna(0)
    .reindex(columns=all_items, fill_value=0)
)

# ── Fit KNN on train set ──────────────────────────────────────────────────────
print("Fitting KNN...")
knn = NearestNeighbors(n_neighbors=21, metric="cosine", algorithm="brute", n_jobs=-1)
knn.fit(train_matrix.values)

user_means     = train_matrix.replace(0, np.nan).mean(axis=1)
popularity_map = train_df.groupby("item_id")["rating"].count().to_dict()
print("Ready.\n")


# ── Recommendation engine (on train matrix) ───────────────────────────────────
def get_recommendations(user_id: int, n: int = 10, k: int = 20) -> list:
    if user_id not in train_matrix.index:
        return []

    user_vec = train_matrix.loc[user_id].values.reshape(1, -1)
    dists, idxs = knn.kneighbors(user_vec, n_neighbors=k + 1)

    sims      = 1 - dists.flatten()
    neighbors = train_matrix.index[idxs.flatten()]
    similar   = [(int(u), float(s)) for u, s in zip(neighbors, sims) if u != user_id and s > 0]

    user_row      = train_matrix.loc[user_id]
    unseen        = user_row[user_row == 0].index.tolist()
    target_mean   = user_means[user_id]

    scores, sim_sums, counts = {}, {}, {}
    for sim_user, sim in similar:
        sim_row  = train_matrix.loc[sim_user]
        sim_mean = user_means[sim_user]
        for m in unseen:
            r = sim_row[m]
            if r > 0:
                scores[m]   = scores.get(m, 0)   + sim * (r - sim_mean)
                sim_sums[m] = sim_sums.get(m, 0) + abs(sim)
                counts[m]   = counts.get(m, 0)   + 1

    preds = {}
    for m in scores:
        if sim_sums[m] > 0 and counts[m] >= 1:
            pred     = target_mean + scores[m] / sim_sums[m]
            preds[m] = max(1.0, min(5.0, pred))

    return [mid for mid, _ in sorted(preds.items(), key=lambda x: x[1], reverse=True)[:n]]


# ── Metrics ───────────────────────────────────────────────────────────────────
def precision_at_k(user_id, k, threshold):
    liked = set(test_df[(test_df["user_id"] == user_id) & (test_df["rating"] >= threshold)]["item_id"])
    if not liked:
        return 0.0
    recs = set(get_recommendations(user_id, n=k))
    return len(recs & liked) / k if recs else 0.0


def recall_at_k(user_id, k, threshold):
    liked = set(test_df[(test_df["user_id"] == user_id) & (test_df["rating"] >= threshold)]["item_id"])
    if not liked:
        return 0.0
    recs = set(get_recommendations(user_id, n=k))
    return len(recs & liked) / len(liked) if recs else 0.0


def f1_at_k(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def catalog_coverage(n_users, k):
    users = train_matrix.index[:n_users].tolist()
    seen  = set()
    for u in users:
        seen.update(get_recommendations(u, n=k))
    return round(len(seen) / len(all_items), 4)


def avg_popularity(user_id, k):
    recs = get_recommendations(user_id, n=k)
    return np.mean([popularity_map.get(m, 0) for m in recs]) if recs else 0.0


# ── Main evaluation ───────────────────────────────────────────────────────────
def evaluate(n_users=50, k=10, threshold=4.0):
    print(f"{'='*55}")
    print(f"  KNN Movie Recommender — Evaluation Report")
    print(f"{'='*55}")
    print(f"  Users    : {n_users}  |  K : {k}  |  Threshold : ≥{threshold}★")
    print(f"  Split    : 80% train / 20% test per user")
    print(f"{'='*55}\n")

    users = train_matrix.index[:n_users].tolist()
    P, R, F, Pop = [], [], [], []
    t0 = time.time()

    for i, uid in enumerate(users):
        if (i + 1) % 10 == 0:
            print(f"  Evaluating user {i+1}/{n_users}...")
        p = precision_at_k(uid, k, threshold)
        r = recall_at_k(uid, k, threshold)
        P.append(p); R.append(r)
        F.append(f1_at_k(p, r))
        Pop.append(avg_popularity(uid, k))

    elapsed = round(time.time() - t0, 2)
    print(f"\n  Computing coverage...")
    cov = catalog_coverage(n_users, k)

    p_mean = round(float(np.mean(P)), 4)
    r_mean = round(float(np.mean(R)), 4)
    f_mean = round(float(np.mean(F)), 4)
    pop_m  = round(float(np.mean(Pop)), 2)

    print(f"\n{'─'*55}")
    print(f"  RESULTS")
    print(f"{'─'*55}")
    print(f"  Precision@{k}      : {p_mean} ± {round(float(np.std(P)),4)}")
    print(f"  Recall@{k}         : {r_mean} ± {round(float(np.std(R)),4)}")
    print(f"  F1@{k}             : {f_mean}")
    print(f"  Catalog Coverage  : {cov:.1%}  ({int(cov*len(all_items))} / {len(all_items)} movies)")
    print(f"  Avg Popularity    : {pop_m} ratings/movie")
    print(f"  Eval Time         : {elapsed}s")
    print(f"{'─'*55}")

    print(f"\n  INTERPRETATION")
    print(f"{'─'*55}")
    print(f"  {'✅' if p_mean >= 0.10 else '⚠️ ' if p_mean >= 0.05 else '❌'} Precision@{k} = {p_mean}  {'Good' if p_mean >= 0.10 else 'Average — normal for user-based CF' if p_mean >= 0.05 else 'Low — consider SVD or item-based CF'}")
    print(f"  {'✅' if cov >= 0.20 else '⚠️ '} Coverage = {cov:.1%}  {'Good diversity' if cov >= 0.20 else 'Popularity bias present'}")
    print(f"  {'✅' if pop_m <= 200 else '⚠️ '} Avg popularity = {pop_m}  {'Diverse recommendations' if pop_m <= 200 else 'Favoring blockbusters'}")
    print(f"{'='*55}\n")

    return {"precision": p_mean, "recall": r_mean, "f1": f_mean, "coverage": cov, "avg_popularity": pop_m}


# ── Compare K values ──────────────────────────────────────────────────────────
def compare_k_values(k_values=[5, 10, 20, 50], n_users=30, threshold=4.0):
    print(f"\n{'='*55}")
    print(f"  K Comparison (n_users={n_users}, threshold≥{threshold})")
    print(f"{'='*55}")
    print(f"  {'K':<6} {'Precision':<14} {'Recall':<14} {'F1'}")
    print(f"  {'─'*48}")
    users = train_matrix.index[:n_users].tolist()
    for k in k_values:
        P = [precision_at_k(u, k, threshold) for u in users]
        R = [recall_at_k(u, k, threshold)    for u in users]
        p = round(float(np.mean(P)), 4)
        r = round(float(np.mean(R)), 4)
        print(f"  {k:<6} {p:<14} {r:<14} {round(f1_at_k(p,r),4)}")
    print(f"{'='*55}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",         type=int,   default=10)
    parser.add_argument("--users",     type=int,   default=50)
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--compare-k", action="store_true")
    args = parser.parse_args()

    evaluate(n_users=args.users, k=args.k, threshold=args.threshold)

    if args.compare_k:
        compare_k_values(n_users=args.users, threshold=args.threshold)