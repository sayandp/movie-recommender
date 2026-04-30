import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

if not os.path.exists("model/knn_model.pkl"):
    print("Model not found — training now...")
    from model.train import train_model
    train_model()

from app.gradio_ui import build_demo

demo = build_demo()
demo.launch(server_name="0.0.0.0", server_port=7860)