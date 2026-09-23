from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from api.service import list_models

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
async def get_models() -> list[dict[str, Any]]:
    return list_models()
