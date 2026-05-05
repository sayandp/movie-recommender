import sys
import os

# Add project root to path so 'app' folder is found correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Train model if not already trained
if not os.path.exists("model/knn_model.pkl"):
    print("Model not found — training now...")
    import subprocess
    subprocess.run([sys.executable, "model/train.py"], check=True)

from app.gradio_ui import build_demo

demo = build_demo()
demo.launch(server_name="0.0.0.0", server_port=7860)