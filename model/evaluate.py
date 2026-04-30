import numpy as np
from app.data_loader import load_ratings
from app.recommender import recommend_movie_ids


def evaluate(k=10, n_users=50):
    print("Loading data...")
    df = load_ratings()

    # Build user → liked movies (rating ≥ 4)
    user_likes = {}

    for user_id in df["user_id"].unique():
        liked = df[(df["user_id"] == user_id) & (df["rating"] >= 4)]["item_id"].tolist()
        if len(liked) >= 5:  # ensure meaningful evaluation
            user_likes[user_id] = liked

    users = list(user_likes.keys())[:n_users]

    scores = []

    print("Evaluating...")

    for user in users:
        try:
            rec_ids = recommend_movie_ids(user, n=k)
            liked = set(user_likes[user])

            hits = sum(1 for m in rec_ids if m in liked)

            score = hits / k
            scores.append(score)

        except:
            continue

    print(f"Precision@{k}: {np.mean(scores):.4f}")


if __name__ == "__main__":
    evaluate()