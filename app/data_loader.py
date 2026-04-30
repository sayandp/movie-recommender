import pandas as pd
import os

DATA_DIR      = "data/ml-100k"
RATINGS_PATH  = os.path.join(DATA_DIR, "u.data")
MOVIES_PATH   = os.path.join(DATA_DIR, "u.item")


def load_ratings() -> pd.DataFrame:
    """Load MovieLens 100K ratings → user_id, item_id, rating."""
    if not os.path.exists(RATINGS_PATH):
        raise FileNotFoundError(
            f"Ratings file not found at '{RATINGS_PATH}'.\n"
            "Make sure ml-100k is extracted inside data/ folder."
        )
    df = pd.read_csv(
        RATINGS_PATH,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    return df.drop(columns=["timestamp"])


def load_movies() -> pd.DataFrame:
    """Load MovieLens 100K movie titles → item_id, title."""
    if not os.path.exists(MOVIES_PATH):
        raise FileNotFoundError(
            f"Movies file not found at '{MOVIES_PATH}'.\n"
            "Make sure ml-100k is extracted inside data/ folder."
        )
    df = pd.read_csv(
        MOVIES_PATH,
        sep="|",
        names=[
            "item_id", "title", "release_date", "video_release_date",
            "imdb_url", "unknown", "Action", "Adventure", "Animation",
            "Children", "Comedy", "Crime", "Documentary", "Drama",
            "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
            "Romance", "Sci-Fi", "Thriller", "War", "Western",
        ],
        encoding="latin-1",
        usecols=["item_id", "title", "release_date"],
    )
    return df


def load_all():
    """Merge ratings with movie titles."""
    ratings = load_ratings()
    movies  = load_movies()
    merged  = ratings.merge(movies[["item_id", "title"]], on="item_id", how="left")
    return merged, movies


if __name__ == "__main__":
    ratings = load_ratings()
    movies  = load_movies()
    print(f"Ratings : {len(ratings):,}")
    print(f"Users   : {ratings['user_id'].nunique():,}")
    print(f"Movies  : {ratings['item_id'].nunique():,}")
    print(ratings.head())
    print(movies.head())