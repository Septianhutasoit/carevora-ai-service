import os
import pandas as pd
from sentence_transformers import InputExample

def generate_default_dataset(csv_path: str):
    """
    Membuat dataset latih awal otomatis jika berkas CSV belum ada di komputer Anda.
    Berisi pasangan (anchor/text, label/career) untuk melatih relasi kesamaan vektor.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    data = {
        "text": [
            "php laravel mysql docker linux git postgresql redis nestjs nodejs backend",
            "laravel php sql docker api server backend development postman",
            "nestjs typescript postgresql docker nodejs backend rest-api microservices",
            
            "react next.js tailwind css html javascript typescript frontend web",
            "reactjs nextjs tailwindcss css3 html5 frontend ui developer redux integration",
            "next.js react tailwind typescript html css frontend responsive",
            
            "docker linux kubernetes aws ci/cd git bash automation devops cloud",
            "devops docker kubernetes linux bash aws cicd pipeline github-actions",
            "aws azure cloud kubernetes docker linux infrastructure monitoring devops"
        ],
        "label": [
            "Backend Developer", "Backend Developer", "Backend Developer",
            "Frontend Developer", "Frontend Developer", "Frontend Developer",
            "DevOps Engineer", "DevOps Engineer", "DevOps Engineer"
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Dataset latih berhasil digenerate di: {csv_path}")

def load_training_data(csv_path: str) -> list[InputExample]:
    """
    Membaca CSV dan mengonversinya menjadi format InputExample SentenceTransformers.
    """
    if not os.path.exists(csv_path):
        generate_default_dataset(csv_path)
        
    df = pd.read_csv(csv_path)
    examples = []
    
    for _, row in df.iterrows():
        examples.append(InputExample(texts=[str(row["text"]), str(row["label"])]))
        
    return examples