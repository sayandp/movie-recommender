import pickle
import numpy as np
import pandas as pd
import math
import time
from typing import List, Tuple, Dict, Optional

from app.data_loader import load_movies, load_ratings

MODEL_PATH  = "model/knn_model.pkl"
MATRIX_PATH = "model/user_item_matrix.pkl"

# ── Load model + matrix ───────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(MATRIX_PATH, "rb") as f:
    user_item: pd.DataFrame = pickle.load(f)

# ── Movie metadata ────────────────────────────────────────────────────────────
movies_df    = load_movies()
movie_titles = dict(zip(movies_df["item_id"], movies_df["title"]))

# ── Popularity map ────────────────────────────────────────────────────────────
ratings_df     = load_ratings()
popularity_map: Dict[int, int] = (
    ratings_df.groupby("item_id")["rating"].count().to_dict()
)

# ── Cold start: Bayesian average top-50 ──────────────────────────────────────
_COLD_START_MIN = 50


def _build_popular_movies(n: int = 50) -> List[Tuple[int, float]]:
    stats = (
        ratings_df.groupby("item_id")["rating"]
        .agg(["count", "mean"])
        .rename(columns={"count": "n", "mean": "R"})
    )
    C = stats["R"].mean()
    m = _COLD_START_MIN
    stats["bayesian"] = (
        (stats["n"] / (stats["n"] + m)) * stats["R"]
        + (m / (stats["n"] + m)) * C
    )
    top = stats[stats["n"] >= m].nlargest(n, "bayesian")
    return list(zip(top.index.tolist(), top["bayesian"].round(3).tolist()))


_popular_movies: List[Tuple[int, float]] = _build_popular_movies(50)

# ── Cache ─────────────────────────────────────────────────────────────────────
_rec_cache: Dict[tuple, list] = {}
_CACHE_MAX   = 500
_cache_hits  = 0
_cache_miss  = 0


def cache_stats() -> Dict:
    return {
        "cached_users": len(_rec_cache),
        "hits":         _cache_hits,
        "misses":       _cache_miss,
        "hit_rate":     round(_cache_hits / max(1, _cache_hits + _cache_miss), 3),
    }


def invalidate_cache(user_id: Optional[int] = None):
    global _rec_cache
    if user_id is None:
        _rec_cache = {}
    else:
        _rec_cache = {k: v for k, v in _rec_cache.items() if k[0] != user_id}


# ── Similar users ─────────────────────────────────────────────────────────────
def get_similar_users(user_id: int, k: int = 20) -> List[Tuple[int, float]]:
    if user_id not in user_item.index:
        raise ValueError(f"User {user_id} not found in training data.")

    user_vector = user_item.loc[user_id].values.reshape(1, -1)
    distances, indices = model.kneighbors(user_vector, n_neighbors=k + 1)

    similarities = 1 - distances.flatten()
    neighbors    = user_item.index[indices.flatten()]

    return [
        (int(u), float(sim))
        for u, sim in zip(neighbors, similarities)
        if u != user_id and sim > 0
    ]


# ── Core prediction ───────────────────────────────────────────────────────────
def _compute_predictions(
    user_id: int,
    n: int = 10,
    k_neighbors: int = 20,
    popularity_penalty: bool = True,
) -> List[Tuple[int, float]]:

    similar_users = get_similar_users(user_id, k=k_neighbors)
    user_ratings  = user_item.loc[user_id]
    unseen_movies = user_ratings[user_ratings == 0].index.tolist()

    user_means  = user_item.replace(0, np.nan).mean(axis=1)
    target_mean = user_means[user_id]

    scores:        Dict[int, float] = {}
    sim_sums:      Dict[int, float] = {}
    rating_counts: Dict[int, int]   = {}

    for sim_user, sim in similar_users:
        sim_ratings    = user_item.loc[sim_user]
        sim_user_mean  = user_means[sim_user]

        for movie in unseen_movies:
            raw = sim_ratings[movie]
            if raw > 0:
                adjusted             = raw - sim_user_mean
                scores[movie]        = scores.get(movie, 0)        + sim * adjusted
                sim_sums[movie]      = sim_sums.get(movie, 0)      + abs(sim)
                rating_counts[movie] = rating_counts.get(movie, 0) + 1

    predictions: Dict[int, float] = {}
    for movie in scores:
        if sim_sums[movie] > 0 and rating_counts[movie] >= 2:
            pred = target_mean + (scores[movie] / sim_sums[movie])
            pred = max(1.0, min(5.0, pred))

            if popularity_penalty:
                n_ratings = popularity_map.get(movie, 1)
                pred      = pred / math.log1p(n_ratings)

            predictions[movie] = round(pred, 4)

    return sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]


# ── Public: recommend with titles ────────────────────────────────────────────
def recommend_movies(
    user_id: int,
    n: int = 10,
    popularity_penalty: bool = True,
) -> List[Tuple[str, float]]:
    """
    Returns list of (movie_title, predicted_score).
    Falls back to popular movies for unknown users (cold start).
    Results are cached after first call.
    """
    global _cache_hits, _cache_miss

    # Cold start
    if user_id not in user_item.index:
        return [
            (movie_titles.get(mid, "Unknown"), score)
            for mid, score in _popular_movies[:n]
        ]

    # Cache lookup
    cache_key = (user_id, n, popularity_penalty)
    if cache_key in _rec_cache:
        _cache_hits += 1
        return _rec_cache[cache_key]

    _cache_miss += 1

    top_movies = _compute_predictions(user_id, n, popularity_penalty=popularity_penalty)
    result = [
        (movie_titles.get(mid, "Unknown"), float(score))
        for mid, score in top_movies
    ]

    # Evict oldest if full
    if len(_rec_cache) >= _CACHE_MAX:
        del _rec_cache[next(iter(_rec_cache))]
    _rec_cache[cache_key] = result

    return result


# ── Public: IDs only (for evaluation) ────────────────────────────────────────
def recommend_movie_ids(user_id: int, n: int = 10) -> List[int]:
    if user_id not in user_item.index:
        return [mid for mid, _ in _popular_movies[:n]]
    return [mid for mid, _ in _compute_predictions(user_id, n)]


# ── Evaluation ────────────────────────────────────────────────────────────────
def precision_at_k(user_id: int, k: int = 10, threshold: float = 4.0) -> float:
    liked = set(
        ratings_df[
            (ratings_df["user_id"] == user_id) &
            (ratings_df["rating"] >= threshold)
        ]["item_id"].tolist()
    )
    if not liked:
        return 0.0
    recommended = set(recommend_movie_ids(user_id, n=k))
    return round(len(recommended & liked) / k, 4)


def recall_at_k(user_id: int, k: int = 10, threshold: float = 4.0) -> float:
    liked = set(
        ratings_df[
            (ratings_df["user_id"] == user_id) &
            (ratings_df["rating"] >= threshold)
        ]["item_id"].tolist()
    )
    if not liked:
        return 0.0
    recommended = set(recommend_movie_ids(user_id, n=k))
    return round(len(recommended & liked) / len(liked), 4)


def evaluate_model(n_users: int = 50, k: int = 10, threshold: float = 4.0) -> Dict:
    print(f"Evaluating {n_users} users (k={k}, threshold≥{threshold})...")
    t0 = time.time()

    sample_users = user_item.index[:n_users].tolist()
    precisions   = [precision_at_k(u, k, threshold) for u in sample_users]
    recalls      = [recall_at_k(u, k, threshold)    for u in sample_users]

    results = {
        "n_users_evaluated": n_users,
        "k":                 k,
        "threshold":         threshold,
        "precision@k_mean":  round(float(np.mean(precisions)), 4),
        "precision@k_std":   round(float(np.std(precisions)),  4),
        "recall@k_mean":     round(float(np.mean(recalls)),    4),
        "recall@k_std":      round(float(np.std(recalls)),     4),
        "eval_time_s":       round(time.time() - t0, 2),
    }

    print(f"  Precision@{k}: {results['precision@k_mean']} ± {results['precision@k_std']}")
    print(f"  Recall@{k}   : {results['recall@k_mean']} ± {results['recall@k_std']}")
    return results


if __name__ == "__main__":
    print("=== Known user (ID=1) ===")
    for title, score in recommend_movies(1, n=5):
        print(f"  {score:.3f}  {title}")

    print("\n=== Cold start (ID=9999) ===")
    for title, score in recommend_movies(9999, n=5):
        print(f"  {score:.3f}  {title}")

    print("\n=== Cache test ===")
    t0 = time.time(); recommend_movies(1)
    print(f"  1st call : {round(time.time()-t0, 4)}s")
    t0 = time.time(); recommend_movies(1)
    print(f"  2nd call : {round(time.time()-t0, 6)}s  (cached)")
    print(f"  Stats    : {cache_stats()}")

    print("\n=== Evaluation ===")
    evaluate_model(n_users=30, k=10)