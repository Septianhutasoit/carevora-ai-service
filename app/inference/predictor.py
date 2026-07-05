from app.inference.embeddings import embedder_service
from app.inference.similarity import calculate_cosine_similarity
from app.utils.preprocessing import clean_skills_text

class Predictor:
    def predict_recommendations(self, user_skills: list[str], careers: list[dict]) -> list[dict]:
        # 1. Bersihkan dan buat vektor embedding untuk skill user
        user_text = clean_skills_text(user_skills)
        user_vector = embedder_service.get_embedding(user_text)

        results = []

        # 2. Hitung kecocokan dengan setiap karir yang dikirim oleh NestJS
        for career in careers:
            career_skills = career.get("skills", [])
            career_text = clean_skills_text(career_skills)
            
            # Buat embedding untuk skill pendukung karir
            career_vector = embedder_service.get_embedding(career_text)
            
            # Hitung Cosine Similarity
            score = calculate_cosine_similarity(user_vector, career_vector)
            
            results.append({
                "id": career.get("id"),
                "score": round(score, 4) # Ambil 4 angka di belakang koma
            })

        # 3. Urutkan berdasarkan skor tertinggi (rekomendasi terbaik)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

predictor_service = Predictor()