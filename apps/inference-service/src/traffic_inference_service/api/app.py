from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from .run_inference import load_inference_objects, predict_one, QUESTION_MAP

# =========================
# 설정
# =========================

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./artifacts/final_adapter")

# =========================
# FastAPI 앱 생성
# =========================

app = FastAPI(
    title="Traffic Accident Inference API",
    description="교통사고 영상 기반 과실 추론 API",
    version="1.0.0",
)

model = None
processor = None


# =========================
# 서버 시작 시 모델 1회 로드
# =========================

@app.on_event("startup")
def startup_event():
    global model, processor
    print("[INFO] FastAPI startup: 모델 로드 시작")
    model, processor = load_inference_objects(
        base_model_id=BASE_MODEL_ID,
        adapter_path=ADAPTER_PATH,
    )
    print("[INFO] FastAPI startup: 모델 로드 완료")


# =========================
# 기본 확인용
# =========================

@app.get("/")
def root():
    return {
        "message": "Traffic Accident Inference API is running",
        "available_question_types": list(QUESTION_MAP.keys()),
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None and processor is not None,
    }


# =========================
# 추론 API
# =========================

@app.post("/predict")
async def predict(
    video: UploadFile = File(...),
    question_type: str = Form(...),
):
    global model, processor

    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")

    if question_type not in QUESTION_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 question_type입니다. 지원 목록: {list(QUESTION_MAP.keys())}",
        )

    suffix = Path(video.filename).suffix.lower() if video.filename else ".mp4"
    if suffix not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        raise HTTPException(status_code=400, detail="지원하지 않는 비디오 형식입니다.")

    temp_dir = tempfile.mkdtemp(prefix="traffic_infer_")
    temp_video_path = os.path.join(temp_dir, f"input{suffix}")

    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        answer = predict_one(
            model=model,
            processor=processor,
            video_path=temp_video_path,
            question_type=question_type,
        )

        return JSONResponse(
            content={
                "success": True,
                "question_type": question_type,
                "filename": video.filename,
                "answer": answer,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass