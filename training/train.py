import os
import sys
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses

# Menambahkan root folder ke sys.path agar modul app dapat terdeteksi saat off
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset_loader import load_training_data
from app.config import settings

def start_finetuning():
    print("====================================================")
    print("  MEMULAI PIPELINE FINE-TUNING MODEL AI CAREVORA     ")
    print("====================================================")

    # 1. Tentukan path dataset
    dataset_path = "datasets/processed/train.csv"
    
    # 2. Muat data latih menggunakan Loader kita
    train_examples = load_training_data(dataset_path)
    
    # 3. Masukkan ke PyTorch DataLoader (membagi data ke ukuran batch kecil)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=3)
    
    # 4. Ambil Model BGE-small dasar dari HuggingFace
    print(f"Loading base model: {settings.BASE_MODEL} ...")
    model = SentenceTransformer(settings.BASE_MODEL)
    
    # 5. Tentukan Fungsi Kehilangan (Loss Function) murni riset AI
    # MultipleNegativesRankingLoss sangat ampuh untuk melatih pencocokan teks semantik pendek
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # 6. Jalankan Proses Pelatihan (Fine-Tuning)
    epochs = 4  # Latih sebanyak 4 siklus penuh (Sangat cepat & ringan, < 1 menit di CPU)
    print(f"Memulai training selama {epochs} Epochs...")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=10,
        show_progress_bar=True
    )
    
    # 7. Simpan model hasil fine-tuning secara lokal ke folder saved_models
    os.makedirs(settings.MODEL_PATH, exist_ok=True)
    model.save(settings.MODEL_PATH)
    
    print("====================================================")
    print(f" SUCCESS: Model hasil training disimpan di: {settings.MODEL_PATH}")
    print("====================================================")

if __name__ == "__main__":
    start_finetuning()