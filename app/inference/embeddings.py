import os
from sentence_transformers import SentenceTransformer
from app.config import settings

class Embedder:
    def __init__(self):
        # Memastikan folder model hasil fine-tuning ada DAN berisi file config.json yang valid
        model_config_path = os.path.join(settings.MODEL_PATH, "config.json")
        
        if os.path.exists(model_config_path):
            print(f"Loading fine-tuned model dari: {settings.MODEL_PATH}")
            self.model = SentenceTransformer(settings.MODEL_PATH)
        else:
            print(f"Loading base model dari HuggingFace: {settings.BASE_MODEL}")
            self.model = SentenceTransformer(settings.BASE_MODEL)

    def get_embedding(self, text: str):
        # Menghasilkan vektor embedding yang ternormalisasi (L2 normalization)
        return self.model.encode(text, normalize_embeddings=True)

# Inisialisasi satu kali agar tidak meload model berulang-ulang di setiap request (Singleton)
embedder_service = Embedder()