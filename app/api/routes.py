from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.inference.predictor import predictor_service

router = APIRouter()

# Skema validasi data masuk menggunakan Pydantic (seperti DTO di NestJS)
class CareerItem(BaseModel):
    id: str
    title: str
    skills: list[str]

class RecommendationRequest(BaseModel):
    user_skills: list[str]
    careers: list[CareerItem]

@router.post("/recommend")
async def get_recommendations(payload: RecommendationRequest):
    try:
        if not payload.user_skills:
            raise HTTPException(status_code=400, detail="Daftar skill user tidak boleh kosong")
        if not payload.careers:
            raise HTTPException(status_code=400, detail="Daftar kandidat karir tidak boleh kosong")

        # Konversi skema Pydantic menjadi format list dictionary Python
        careers_dict = [career.model_dump() for career in payload.careers]

        # Jalankan prediksi kecocokan
        recommendations = predictor_service.predict_recommendations(
            user_skills=payload.user_skills,
            careers=careers_dict
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))