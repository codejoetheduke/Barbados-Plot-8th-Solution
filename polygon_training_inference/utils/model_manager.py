from ultralytics import YOLO
from pathlib import Path

class ModelManager:
    @staticmethod
    def load_pretrained(model_name="CodeJoe/barbados-yolov8m-mapper"):
        try:
            from huggingface_hub import hf_hub_download
            print(f"📥 Downloading model from Hugging Face: {model_name}")
            model_path = hf_hub_download(
                repo_id=model_name,
                filename="segment/train/weights/best.pt"
            )
            print(f"✅ Model downloaded to: {model_path}")
            model = YOLO(model_path)
            print("🚀 Model loaded and ready for inference!")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

    @staticmethod
    def load_local(model_path):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return YOLO(model_path)
