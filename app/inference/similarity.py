import numpy as np

def calculate_cosine_similarity(vector_a, vector_b) -> float:
    """
    Menghitung kemiripan kosinus (Cosine Similarity) antara dua vektor.
    Karena model BGE sudah mengeluarkan vektor ternormalisasi (normalize_embeddings=True),
    kita cukup menggunakan Dot Product (perkalian titik) untuk efisiensi komputasi yang tinggi.
    """
    dot_product = np.dot(vector_a, vector_b)
    score = float(dot_product)
    # Menjaga agar skor berada di rentang absolut 0.0 sampai 1.0
    return max(0.0, min(1.0, score))