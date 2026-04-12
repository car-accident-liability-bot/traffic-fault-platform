from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    # 주석: 추후 모델 로딩 상태 점검 필드를 추가하기 쉽도록 단순 dict로 유지한다.
    return {"status": "ok", "service": "traffic-inference-service"}
