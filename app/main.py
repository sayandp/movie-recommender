from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

from app.schemas import (
    RecommendResponse, RecommendationItem,
    SimilarUsersResponse, BatchRequest, BatchResponse,
    EvalResponse, HealthResponse,
)
from app.recommender import (
    recommend_movies, recommend_movie_ids,
    get_similar_users, evaluate_model,
    cache_stats, invalidate_cache,
    user_item,
)
from app.gradio_ui import build_demo

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🎬 Movie Recommender API",
    description="User-based KNN collaborative filtering on MovieLens 100K",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount Gradio at /demo ─────────────────────────────────────────────────────
demo = build_demo()
app  = gr.mount_gradio_app(app, demo, path="/demo")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return {
        "status":      "ok",
        "model":       "KNN (cosine, user-based)",
        "n_users":     int(user_item.shape[0]),
        "n_items":     int(user_item.shape[1]),
        "cache_stats": cache_stats(),
    }


@app.get("/recommend/{user_id}", response_model=RecommendResponse, tags=["Recommend"])
def recommend(
    user_id: int,
    n: int = Query(default=10, ge=1, le=50, description="Number of recommendations"),
    popularity_penalty: bool = Query(default=True, description="Reduce popular-movie bias"),
):
    is_cold = user_id not in user_item.index
    recs    = recommend_movies(user_id, n=n, popularity_penalty=popularity_penalty)
    return {
        "user_id":       user_id,
        "is_cold_start": is_cold,
        "recommendations": [
            {"title": title, "score": round(score, 4)}
            for title, score in recs
        ],
    }


@app.post("/recommend/batch", response_model=BatchResponse, tags=["Recommend"])
def recommend_batch(request: BatchRequest):
    results = []
    for uid in request.user_ids:
        is_cold = uid not in user_item.index
        recs    = recommend_movies(uid, n=request.n)
        results.append({
            "user_id":       uid,
            "is_cold_start": is_cold,
            "recommendations": [
                {"title": t, "score": round(s, 4)} for t, s in recs
            ],
        })
    return {"results": results}


@app.get("/similar-users/{user_id}", response_model=SimilarUsersResponse, tags=["Explore"])
def similar_users(
    user_id: int,
    k: int = Query(default=10, ge=1, le=50),
):
    if user_id not in user_item.index:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    neighbors = get_similar_users(user_id, k=k)
    return {
        "user_id": user_id,
        "similar_users": [
            {"user_id": int(uid), "similarity": round(sim, 4)}
            for uid, sim in neighbors
        ],
    }


@app.get("/evaluate", response_model=EvalResponse, tags=["System"])
def evaluate(
    n_users:   int   = Query(default=50,  ge=10, le=500),
    k:         int   = Query(default=10,  ge=1,  le=50),
    threshold: float = Query(default=4.0, ge=1,  le=5),
):
    result = evaluate_model(n_users=n_users, k=k, threshold=threshold)
    return {
        "n_users_evaluated": result["n_users_evaluated"],
        "k":                 result["k"],
        "threshold":         result["threshold"],
        "precision_at_k":    result["precision@k_mean"],
        "recall_at_k":       result["recall@k_mean"],
        "eval_time_s":       result["eval_time_s"],
    }


@app.delete("/cache", tags=["System"])
def clear_cache(user_id: int = Query(default=None)):
    invalidate_cache(user_id)
    return {"message": "Cache cleared", "cache_stats": cache_stats()}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)