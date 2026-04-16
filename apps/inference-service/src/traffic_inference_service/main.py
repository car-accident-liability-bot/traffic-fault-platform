from traffic_inference_service.api.app import app

import uvicorn


def run() -> None:
    uvicorn.run(
        "traffic_inference_service.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    run()