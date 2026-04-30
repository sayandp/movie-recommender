---
title: Movie Recommender
emoji: 👁
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
---

# 🎬 Movie Recommender System

User-based KNN Collaborative Filtering on MovieLens 100K.

## Stack
- **Algorithm** : KNN with cosine similarity (sklearn)
- **API**       : FastAPI + Uvicorn
- **UI**        : Gradio
- **Dataset**   : MovieLens 100K (943 users, 1682 movies, 100K ratings)

## Project Structure
movie/
├── app/
│ ├── init.py
│ ├── data_loader.py
│ ├── recommender.py
│ ├── gradio_ui.py
│ ├── main.py
│ └── schemas.py
├── model/
│ ├── train.py
│ ├── knn_model.pkl
│ └── user_item_matrix.pkl
├── data/
│ └── ml-100k/
├── Dockerfile
├── requirements.txt
└── README.md


## Quickstart

```bash
pip install -r requirements.txt
python model/train.py
uvicorn app.main:app --reload --port 8000

Then open:

Gradio UI → http://localhost:8000/demo
API Docs → http://localhost:8000/docs
API Endpoints
Method	Endpoint	Description
GET	/recommend/{user_id}	Top-N recommendations
POST	/recommend/batch	Bulk recommendations
GET	/similar-users/{user_id}	K nearest neighbours
GET	/evaluate	Precision@K + Recall@K
GET	/health	Model + cache status
DELETE	/cache	Clear recommendation cache
Features
Mean-centered KNN
Popularity penalty
Cold start fallback
In-memory caching
Evaluation metrics
Gradio UI + FastAPI

---

## ⚠️ Important inconsistency you should fix
You had:
- HF config: `app_file: app.py`
- Your code: uses **FastAPI (`app.main:app`)**

These are incompatible unless:
- You actually have a top-level `app.py` that launches Gradio

### If your entry point is actually `demo.py` or something else:
Fix this line:
```yaml
app_file: app.py
