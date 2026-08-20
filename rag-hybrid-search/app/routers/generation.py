from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.dependencies import get_generation_service, get_llm_provider
from app.models.schemas import GenerateRequest, GenerateResponse
from app.services.generation_service import GenerationService

router = APIRouter(prefix="/generation", tags=["generation"])


@router.post("/answer", response_model=GenerateResponse)
def generate_answer(
    request: GenerateRequest,
    service: GenerationService = Depends(get_generation_service),
    settings: Settings = Depends(get_settings),
) -> GenerateResponse:
    try:
        llm = get_llm_provider(request.provider, settings=settings)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return service.answer(query=request.query, llm=llm, top_k=request.top_k)
