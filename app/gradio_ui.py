import gradio as gr
import pandas as pd
import json
import requests
import urllib.parse
import re

from app.recommender import (
    recommend_movies,
    get_similar_users,
    evaluate_model,
    cache_stats,
    user_item,
)
from app.data_loader import load_ratings

ratings_df   = load_ratings()
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
POSTER_BASE  = "https://image.tmdb.org/t/p/w300"
FALLBACK     = "https://placehold.co/300x450/e8e8e8/999999?text=🎬"
_cache       = {}


def _clean_title(title):
    m     = re.search(r'\((\d{4})\)', title)
    year  = int(m.group(1)) if m else None
    clean = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
    if   clean.endswith(', The'): clean = 'The '  + clean[:-5]
    elif clean.endswith(', A'):   clean = 'A '    + clean[:-3]
    elif clean.endswith(', An'):  clean = 'An '   + clean[:-4]
    return clean, year


def get_poster(title):
    if title in _cache:
        return _cache[title]
    try:
        clean, year = _clean_title(title)
        q    = urllib.parse.quote(clean)
        url  = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={q}"
        if year: url += f"&year={year}"
        r    = requests.get(url, timeout=3)
        data = r.json()
        if data.get("results"):
            path = data["results"][0].get("poster_path")
            if path:
                _cache[title] = POSTER_BASE + path
                return _cache[title]
    except Exception:
        pass
    _cache[title] = FALLBACK
    return FALLBACK


def build_grid(recs, show_posters=True):
    if not recs:
        return "<p style='color:#666;padding:2rem;font-family:Arial'>No results found.</p>"

    cards = ""
    for i, (title, score) in enumerate(recs):
        poster     = get_poster(title) if show_posters else FALLBACK
        clean, yr  = _clean_title(title)
        match      = min(99, max(60, int(score * 18)))
        bar        = "#21a354" if match >= 75 else "#f5a623" if match >= 60 else "#e50914"
        pct        = int((min(score, 5) / 5) * 100)

        cards += f"""
        <div class="card" style="animation-delay:{i*0.05}s">
            <div class="img-wrap">
                <img src="{poster}" alt="{clean}"
                     onerror="this.src='{FALLBACK}'" loading="lazy"/>
                <div class="img-overlay">
                    <span class="rank-tag">#{i+1}</span>
                    <div class="hover-info">
                        <div class="match" style="color:{bar}">▶ {match}% Match</div>
                        <div class="htitle">{clean}</div>
                        <div class="hyear">{yr or ''}</div>
                        <div class="hbar-wrap"><div class="hbar" style="width:{pct}%;background:{bar}"></div></div>
                    </div>
                </div>
            </div>
            <div class="card-foot">
                <div class="ctitle">{clean}</div>
                <div class="cscore">{'★' * int(round(min(score*1.2,5)))}{'☆' * (5 - int(round(min(score*1.2,5))))}</div>
            </div>
        </div>"""

    return f"""
    <style>
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(130px,1fr)); gap:14px; padding:4px 2px 20px; }}
    .card {{ border-radius:10px; overflow:hidden; background:#fff; box-shadow:0 2px 12px rgba(0,0,0,0.10);
             transition:transform .22s ease,box-shadow .22s ease; animation:fadeUp .35s ease both; cursor:pointer; }}
    .card:hover {{ transform:translateY(-6px) scale(1.04); box-shadow:0 12px 36px rgba(0,0,0,0.18); }}
    @keyframes fadeUp {{ from{{opacity:0;transform:translateY(16px)}} to{{opacity:1;transform:translateY(0)}} }}
    .img-wrap {{ position:relative; width:100%; padding-top:145%; background:#f0f0f0; overflow:hidden; }}
    .img-wrap img {{ position:absolute;inset:0;width:100%;height:100%;object-fit:cover;
                     transition:transform .3s ease; display:block; }}
    .card:hover .img-wrap img {{ transform:scale(1.06); }}
    .img-overlay {{ position:absolute;inset:0;background:linear-gradient(to bottom,rgba(0,0,0,.45) 0%,transparent 30%,transparent 55%,rgba(0,0,0,.80) 100%);
                    display:flex;flex-direction:column;justify-content:space-between;padding:8px;
                    opacity:0;transition:opacity .22s ease; }}
    .card:hover .img-overlay {{ opacity:1; }}
    .rank-tag {{ background:#e50914;color:#fff;font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px;width:fit-content; }}
    .hover-info {{ display:flex;flex-direction:column;gap:3px; }}
    .match {{ font-size:12px;font-weight:700; }}
    .htitle {{ font-size:11px;font-weight:600;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }}
    .hyear {{ font-size:10px;color:#ddd; }}
    .hbar-wrap {{ height:2px;background:rgba(255,255,255,.25);border-radius:99px; }}
    .hbar {{ height:100%;border-radius:99px;transition:width .5s ease; }}
    .card-foot {{ padding:8px 10px 10px;background:#fff; }}
    .ctitle {{ font-size:11px;font-weight:600;color:#1a1a1a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:3px; }}
    .cscore {{ font-size:11px;color:#f5a623;letter-spacing:1px; }}
    </style>
    <div class="grid">{cards}</div>"""


def get_recs(user_id, n, penalty, show_posters):
    try:
        uid     = int(user_id)
        is_cold = uid not in user_item.index
        recs    = recommend_movies(uid, n=int(n), popularity_penalty=bool(penalty))
        label   = "⚠️ Unknown user — showing popular picks" if is_cold else f"🎬 **{len(recs)} movies for User {uid}**"
        return label, build_grid(recs, show_posters=bool(show_posters))
    except Exception as e:
        return f"❌ {e}", ""


def get_sim_users(user_id, k):
    try:
        uid = int(user_id)
        if uid not in user_item.index:
            return f"❌ User {uid} not found.", pd.DataFrame()
        nb = get_similar_users(uid, k=int(k))
        df = pd.DataFrame([(i+1, u, round(s,4)) for i,(u,s) in enumerate(nb)], columns=["#","User ID","Similarity"])
        return f"✅ Top {k} users similar to User {uid}", df
    except Exception as e:
        return f"❌ {e}", pd.DataFrame()


def run_eval(n_users, k, threshold):
    try:
        r  = evaluate_model(n_users=int(n_users), k=int(k), threshold=threshold)
        p  = r['precision@k_mean']
        rc = r['recall@k_mean']
        v  = "✅ Good" if p >= 0.10 else "⚠️ Average — normal for KNN" if p >= 0.05 else "❌ Low — consider SVD"
        return f"""### 📊 Results — K={k}, threshold ≥{threshold}★

| Metric | Score |
|--------|-------|
| **Precision@{k}** | `{p}` ± {r['precision@k_std']} |
| **Recall@{k}** | `{rc}` ± {r['recall@k_std']} |
| **Users** | {r['n_users_evaluated']} |
| **Time** | {r['eval_time_s']}s |

**Verdict:** {v}"""
    except Exception as e:
        return f"❌ {e}"


CSS = """
* { box-sizing: border-box; }
body, .gradio-container {
    background: #f3f4f6 !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
    color: #111 !important;
}
.gradio-container { max-width: 1300px !important; margin: 0 auto !important; padding: 0 !important; }

/* HEADER */
.rm-header {
    background: linear-gradient(135deg, #141414 0%, #1a1a2e 100%);
    padding: 20px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 4px solid #e50914;
}
.rm-logo { font-size: 32px; font-weight: 900; color: #e50914; letter-spacing: -1px; }
.rm-logo span { color: #fff; }
.rm-sub { font-size: 11px; color: #888; letter-spacing: 2px; text-transform: uppercase; margin-top: 2px; }
.rm-stats { display: flex; gap: 24px; }
.rm-stat { text-align: center; }
.rm-stat-n { font-size: 22px; font-weight: 800; color: #e50914; line-height: 1; }
.rm-stat-l { font-size: 9px; color: #888; letter-spacing: 2px; text-transform: uppercase; }

/* TABS */
.tab-nav { background: #fff !important; border-bottom: 2px solid #e5e7eb !important; padding: 0 24px !important; }
.tab-nav button {
    color: #6b7280 !important; font-weight: 600 !important; font-size: 12px !important;
    letter-spacing: 1px !important; text-transform: uppercase !important;
    border-radius: 0 !important; border-bottom: 3px solid transparent !important;
    background: transparent !important; padding: 14px 18px !important;
    transition: all .2s !important;
}
.tab-nav button.selected { color: #e50914 !important; border-bottom-color: #e50914 !important; }
.tab-nav button:hover { color: #111 !important; }

/* SIDEBAR PANEL */
.gr-panel, .gr-box, .gr-form {
    background: #fff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06) !important;
}

/* LABELS */
label {
    color: #6b7280 !important; font-size: 10px !important; font-weight: 700 !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
}

/* NUMBER INPUT */
input[type=number] {
    background: #f9fafb !important; border: 2px solid #e5e7eb !important;
    color: #111 !important; border-radius: 8px !important; font-size: 18px !important;
    font-weight: 700 !important;
}
input[type=number]:focus { border-color: #e50914 !important; box-shadow: 0 0 0 3px rgba(229,9,20,0.1) !important; }

/* SLIDER */
input[type=range] { accent-color: #e50914 !important; }

/* CHECKBOX */
input[type=checkbox] { accent-color: #e50914 !important; transform: scale(1.2); }

/* BUTTON */
button.primary, .gr-button-primary {
    background: #e50914 !important; border: none !important; color: #fff !important;
    font-weight: 800 !important; font-size: 13px !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    border-radius: 8px !important; padding: 14px !important;
    box-shadow: 0 4px 14px rgba(229,9,20,0.30) !important;
    transition: all .2s !important;
}
button.primary:hover { background: #c00812 !important; transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(229,9,20,0.4) !important; }

/* CONTENT AREA */
.content-area { background: #f3f4f6 !important; padding: 16px !important; border-radius: 12px !important; min-height: 400px; }

/* TABLE */
table { background: #fff !important; border-radius: 8px !important; overflow: hidden !important; }
th { background: #f9fafb !important; color: #e50914 !important; font-size: 10px !important;
     letter-spacing: 2px !important; text-transform: uppercase !important; font-weight: 700 !important; padding: 12px !important; }
td { color: #374151 !important; padding: 10px 14px !important; font-size: 13px !important; border-bottom: 1px solid #f3f4f6 !important; }
tr:hover td { background: #fef2f2 !important; }

/* MARKDOWN */
.gr-markdown p, .gr-markdown li { color: #374151 !important; font-size: 13px !important; }
.gr-markdown h3 { color: #e50914 !important; font-weight: 800 !important; }
.gr-markdown code { background: #fef2f2 !important; color: #e50914 !important; border-radius: 4px !important; padding: 1px 6px !important; }
.gr-markdown table { width: 100% !important; }
.gr-markdown td, .gr-markdown th { border: 1px solid #e5e7eb !important; }
"""


def build_demo():
    with gr.Blocks(title="RecoMovies", css=CSS) as demo:

        gr.HTML(f"""
        <div class="rm-header">
            <div>
                <div class="rm-logo">RECO<span>MOVIES</span></div>
                <div class="rm-sub">AI · KNN Collaborative Filtering · MovieLens 100K</div>
            </div>
            <div class="rm-stats">
                <div class="rm-stat"><div class="rm-stat-n">{user_item.shape[0]}</div><div class="rm-stat-l">Users</div></div>
                <div class="rm-stat"><div class="rm-stat-n">{user_item.shape[1]}</div><div class="rm-stat-l">Movies</div></div>
                <div class="rm-stat"><div class="rm-stat-n">100K</div><div class="rm-stat-l">Ratings</div></div>
            </div>
        </div>
        """)

        with gr.Tabs():

            # ── DISCOVER ──────────────────────────────────────────────────────
            with gr.Tab("🎬 DISCOVER"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=220):
                        gr.HTML("<div style='height:4px'></div>")
                        uid_in   = gr.Number(label="User ID  (1 – 943)", value=1, precision=0, minimum=1, maximum=943)
                        n_in     = gr.Slider(label="Movies to show", minimum=4, maximum=48, value=12, step=4)
                        pen_in   = gr.Checkbox(label="Popularity penalty", value=False)
                        post_in  = gr.Checkbox(label="Show posters (needs internet)", value=True)
                        gr.HTML("<div style='height:6px'></div>")
                        rec_btn  = gr.Button("▶  GET RECOMMENDATIONS", variant="primary")

                    with gr.Column(scale=4):
                        rec_lbl  = gr.Markdown("*Enter a User ID (1–943) and click GET RECOMMENDATIONS*")
                        rec_html = gr.HTML()

                rec_btn.click(fn=get_recs, inputs=[uid_in, n_in, pen_in, post_in], outputs=[rec_lbl, rec_html])

            # ── SIMILAR USERS ─────────────────────────────────────────────────
            with gr.Tab("👥 SIMILAR USERS"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=220):
                        gr.HTML("<div style='height:4px'></div>")
                        s_uid = gr.Number(label="User ID  (1 – 943)", value=1, precision=0, minimum=1, maximum=943)
                        s_k   = gr.Slider(label="Number of similar users", minimum=1, maximum=50, value=10, step=1)
                        gr.HTML("<div style='height:6px'></div>")
                        s_btn = gr.Button("▶  FIND SIMILAR USERS", variant="primary")
                    with gr.Column(scale=4):
                        s_lbl = gr.Markdown()
                        s_tbl = gr.Dataframe(headers=["#", "User ID", "Similarity"], interactive=False)

                s_btn.click(fn=get_sim_users, inputs=[s_uid, s_k], outputs=[s_lbl, s_tbl])

            # ── EVALUATION ────────────────────────────────────────────────────
            with gr.Tab("📊 EVALUATION"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=220):
                        gr.HTML("<div style='height:4px'></div>")
                        e_users = gr.Slider(label="Users to evaluate", minimum=10, maximum=200, value=50, step=10)
                        e_k     = gr.Slider(label="K  (list size)", minimum=1, maximum=50, value=10, step=1)
                        e_thr   = gr.Slider(label="Like threshold ★", minimum=1, maximum=5, value=4.0, step=0.5)
                        gr.HTML("<div style='height:6px'></div>")
                        e_btn   = gr.Button("▶  RUN EVALUATION", variant="primary")
                    with gr.Column(scale=4):
                        e_out = gr.Markdown()

                e_btn.click(fn=run_eval, inputs=[e_users, e_k, e_thr], outputs=e_out)

            # ── SYSTEM ────────────────────────────────────────────────────────
            with gr.Tab("⚙️ SYSTEM"):
                gr.HTML(f"""
                <div style="display:flex;gap:14px;flex-wrap:wrap;padding:12px 0 20px">
                    <div style="background:#fff;border:1px solid #e5e7eb;border-top:4px solid #e50914;border-radius:10px;padding:16px 24px;flex:1;min-width:100px;text-align:center">
                        <div style="font-size:28px;font-weight:900;color:#e50914">{user_item.shape[0]}</div>
                        <div style="font-size:10px;color:#9ca3af;letter-spacing:2px;text-transform:uppercase;margin-top:4px">Users</div>
                    </div>
                    <div style="background:#fff;border:1px solid #e5e7eb;border-top:4px solid #e50914;border-radius:10px;padding:16px 24px;flex:1;min-width:100px;text-align:center">
                        <div style="font-size:28px;font-weight:900;color:#e50914">{user_item.shape[1]}</div>
                        <div style="font-size:10px;color:#9ca3af;letter-spacing:2px;text-transform:uppercase;margin-top:4px">Movies</div>
                    </div>
                    <div style="background:#fff;border:1px solid #e5e7eb;border-top:4px solid #e50914;border-radius:10px;padding:16px 24px;flex:1;min-width:100px;text-align:center">
                        <div style="font-size:28px;font-weight:900;color:#e50914">100K</div>
                        <div style="font-size:10px;color:#9ca3af;letter-spacing:2px;text-transform:uppercase;margin-top:4px">Ratings</div>
                    </div>
                    <div style="background:#fff;border:1px solid #e5e7eb;border-top:4px solid #e50914;border-radius:10px;padding:16px 24px;flex:1;min-width:100px;text-align:center">
                        <div style="font-size:28px;font-weight:900;color:#e50914">KNN</div>
                        <div style="font-size:10px;color:#9ca3af;letter-spacing:2px;text-transform:uppercase;margin-top:4px">Algorithm</div>
                    </div>
                    <div style="background:#fff;border:1px solid #e5e7eb;border-top:4px solid #e50914;border-radius:10px;padding:16px 24px;flex:1;min-width:100px;text-align:center">
                        <div style="font-size:28px;font-weight:900;color:#e50914">93.7%</div>
                        <div style="font-size:10px;color:#9ca3af;letter-spacing:2px;text-transform:uppercase;margin-top:4px">Sparsity</div>
                    </div>
                    <div style="background:#fff;border:1px solid #e5e7eb;border-top:4px solid #e50914;border-radius:10px;padding:16px 24px;flex:1;min-width:100px;text-align:center">
                        <div style="font-size:28px;font-weight:900;color:#e50914">cosine</div>
                        <div style="font-size:10px;color:#9ca3af;letter-spacing:2px;text-transform:uppercase;margin-top:4px">Similarity</div>
                    </div>
                </div>
                """)
                c_btn = gr.Button("▶  REFRESH CACHE", variant="primary")
                c_out = gr.Code(label="Cache Stats", language="json")
                c_btn.click(fn=lambda: json.dumps(cache_stats(), indent=2), outputs=c_out)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)