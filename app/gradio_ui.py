import gradio as gr
import pandas as pd
import json

from app.recommender import (
    recommend_movies,
    get_similar_users,
    evaluate_model,
    cache_stats,
    user_item,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _recs_to_df(recs):
    return pd.DataFrame(
        [(i + 1, title, round(score, 4)) for i, (title, score) in enumerate(recs)],
        columns=["#", "Movie Title", "Predicted Score"],
    )


def _similar_to_df(neighbors):
    return pd.DataFrame(
        [(i + 1, uid, round(sim, 4)) for i, (uid, sim) in enumerate(neighbors)],
        columns=["#", "User ID", "Similarity"],
    )


# ── Tab functions ─────────────────────────────────────────────────────────────
def get_recommendations(user_id: int, n: int, penalty: bool):
    try:
        is_cold = user_id not in user_item.index
        recs    = recommend_movies(user_id, n=n, popularity_penalty=penalty)
        label   = (
            f"⚠️ User {user_id} not found — showing popular movies instead"
            if is_cold
            else f"✅ Top {n} recommendations for User {user_id}"
        )
        return label, _recs_to_df(recs)
    except Exception as e:
        return f"❌ Error: {e}", pd.DataFrame()


def get_similar_users_ui(user_id: int, k: int):
    try:
        if user_id not in user_item.index:
            return f"❌ User {user_id} not found.", pd.DataFrame()
        neighbors = get_similar_users(user_id, k=k)
        return f"✅ Top {k} similar users for User {user_id}", _similar_to_df(neighbors)
    except Exception as e:
        return f"❌ Error: {e}", pd.DataFrame()


def run_evaluation(n_users: int, k: int, threshold: float):
    try:
        result = evaluate_model(n_users=int(n_users), k=int(k), threshold=threshold)
        summary = (
            f"**Precision@{k}**: {result['precision@k_mean']} ± {result['precision@k_std']}\n\n"
            f"**Recall@{k}**:    {result['recall@k_mean']} ± {result['recall@k_std']}\n\n"
            f"**Users evaluated**: {result['n_users_evaluated']}\n\n"
            f"**Time**: {result['eval_time_s']}s"
        )
        return summary
    except Exception as e:
        return f"❌ Error: {e}"


def get_cache_stats():
    stats = cache_stats()
    return json.dumps(stats, indent=2)


# ── Build demo ────────────────────────────────────────────────────────────────
def build_demo() -> gr.Blocks:
    # NEW
    with gr.Blocks(title="🎬 Movie Recommender") as demo:

        gr.Markdown(
            """
            # 🎬 Movie Recommender System
            **User-based KNN Collaborative Filtering** on MovieLens 100K  
            Valid user IDs: **1 – 943** · Valid ratings: **1–5**
            """
        )

        with gr.Tabs():

            # ── Tab 1: Recommendations ────────────────────────────────────
            with gr.Tab("🎯 Recommendations"):
                with gr.Row():
                    with gr.Column(scale=1):
                        user_input  = gr.Number(label="User ID", value=1, precision=0, minimum=1, maximum=943)
                        n_input     = gr.Slider(label="Number of recommendations", minimum=1, maximum=50, value=10, step=1)
                        pen_input   = gr.Checkbox(label="Apply popularity penalty", value=True)
                        rec_btn     = gr.Button("Get Recommendations", variant="primary")
                    with gr.Column(scale=2):
                        rec_label   = gr.Markdown()
                        rec_table   = gr.Dataframe(
                            headers=["#", "Movie Title", "Predicted Score"],
                            interactive=False,
                        )

                rec_btn.click(
                    fn=get_recommendations,
                    inputs=[user_input, n_input, pen_input],
                    outputs=[rec_label, rec_table],
                )

            # ── Tab 2: Similar Users ──────────────────────────────────────
            with gr.Tab("👥 Similar Users"):
                with gr.Row():
                    with gr.Column(scale=1):
                        sim_user_input = gr.Number(label="User ID", value=1, precision=0, minimum=1, maximum=943)
                        sim_k_input    = gr.Slider(label="Number of similar users", minimum=1, maximum=50, value=10, step=1)
                        sim_btn        = gr.Button("Find Similar Users", variant="primary")
                    with gr.Column(scale=2):
                        sim_label = gr.Markdown()
                        sim_table = gr.Dataframe(
                            headers=["#", "User ID", "Similarity"],
                            interactive=False,
                        )

                sim_btn.click(
                    fn=get_similar_users_ui,
                    inputs=[sim_user_input, sim_k_input],
                    outputs=[sim_label, sim_table],
                )

            # ── Tab 3: Evaluation ─────────────────────────────────────────
            with gr.Tab("📊 Evaluation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        eval_users     = gr.Slider(label="Users to evaluate", minimum=10, maximum=200, value=50, step=10)
                        eval_k         = gr.Slider(label="K (list size)", minimum=1, maximum=50, value=10, step=1)
                        eval_threshold = gr.Slider(label="Relevance threshold (min rating)", minimum=1, maximum=5, value=4.0, step=0.5)
                        eval_btn       = gr.Button("Run Evaluation", variant="primary")
                    with gr.Column(scale=2):
                        eval_output = gr.Markdown()

                eval_btn.click(
                    fn=run_evaluation,
                    inputs=[eval_users, eval_k, eval_threshold],
                    outputs=eval_output,
                )

            # ── Tab 4: System Info ────────────────────────────────────────
            with gr.Tab("⚙️ System"):
                cache_btn    = gr.Button("Refresh Cache Stats")
                cache_output = gr.Code(label="Cache Stats", language="json")

                cache_btn.click(fn=get_cache_stats, outputs=cache_output)

                gr.Markdown(
                    f"""
                    ### Dataset Info
                    - **Dataset**: MovieLens 100K  
                    - **Users**: {user_item.shape[0]:,}  
                    - **Movies**: {user_item.shape[1]:,}  
                    - **Algorithm**: KNN with cosine similarity (user-based)  
                    - **API Docs**: [/docs](/docs)
                    """
                )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_port=7860, share=False, theme=gr.themes.Soft(primary_hue="blue"))