from fastapi import APIRouter
from app.models.schema import NewsRequest
from app.services.model_service import predict

router = APIRouter()

@router.post("/predict")
def predict_news(req: NewsRequest):
    return predict(req.text)