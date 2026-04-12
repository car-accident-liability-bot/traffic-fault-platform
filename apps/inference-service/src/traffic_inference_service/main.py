from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from traffic_inference_service.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="traffic-inference-service",
        version="0.1.0",
        description="Inference service skeleton for the traffic fault platform",
    )
    app.include_router(router)
    return app


app = create_app()


def run() -> None:
    uvicorn.run(
        "traffic_inference_service.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    # 주석:
    # - README에 있는 `python -m traffic_inference_service.main` 실행 예시가
    #   실제로도 uvicorn 서버를 띄우도록 진입점을 맞춘다.
    run()
