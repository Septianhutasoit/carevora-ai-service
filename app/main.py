import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.config import settings

app = FastAPI(
    title="Carevora AI Service",
    description="Mesin komputasi Cosine Similarity untuk rekomendasi karir",
    version="1.0.0"
)

# Izinkan komunikasi lintas port (CORS) dari NestJS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Daftarkan rute API
app.include_router(api_router)

@app.get("/")
def root():
    return {"message": "Carevora AI Service is running successfully"}

if __name__ == "__main__":
    # Jalankan server menggunakan konfigurasi dari settings
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)