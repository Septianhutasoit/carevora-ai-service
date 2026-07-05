import os
from sentence_transformers import SentenceTransformer
from app.config import settings

class Embeddings:
    def __init__(self):
        # Gunakan model hasil fine-tuning jika ada, jika tidak, gunakan base model BGE-small
        if os.path.exists(settings.MODEL_PATH):
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