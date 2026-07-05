import re

def clean_skills_text(skills: list[str]) -> str:
    """
    Pembersihan teks input:
    - Menggabungkan list skill menjadi satu string teks yang dipisahkan spasi
    - Mengubah ke huruf kecil (lowercase)
    - Menghapus karakter non-alfanumerik kecuali simbol standar pemrograman (. , + # -)
    """
    text = " ".join(skills).lower()
    # Mengizinkan karakter standar seperti . (Next.js), # (C#), + (C++), dan - (React-Native)
    text = re.sub(r'[^a-z0-9\s\.\-\+#]', '', text)
    return text.strip()