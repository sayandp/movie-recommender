from pydantic import BaseModel, Field
from typing import List, Optional


class RecommendationItem(BaseModel):
    title: str
    score: float


class RecommendResponse(BaseModel):
    user_id:     int
    is_cold_start: bool
    recommendations: List[RecommendationItem]


class SimilarUsersResponse(BaseModel):
    user_id:      int
    similar_users: List[dict]


class BatchRequest(BaseModel):
    user_ids: List[int] = Field(..., example=[1, 2, 3])
    n:        int       = Field(default=10, ge=1, le=50)


class BatchResponse(BaseModel):
    results: List[RecommendResponse]


class EvalResponse(BaseModel):
    n_users_evaluated: int
    k:                 int
    threshold:         float
    precision_at_k:    float
    recall_at_k:       float
    eval_time_s:       float


class HealthResponse(BaseModel):
    status:     str
    model:      str
    n_users:    int
    n_items:    int
    cache_stats: dict