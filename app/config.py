import os
from dotenv import load_dotenv

# Memuat file .env dari folder root
load_dotenv()

class Settings:
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", 8000))
    MODEL_PATH: str = os.getenv("MODEL_PATH", "saved_models/finetuned-career")
    BASE_MODEL: str = os.getenv("BASE_MODEL", "BAAI/bge-small-en-v1.5")

# Instansiasi class Settings ke dalam variabel settings
settings = Settings()