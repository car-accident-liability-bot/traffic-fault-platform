from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoProcessor

from traffic_ai_core.Qwen3_VL_4B_Instruct.model.model import load_model

# =========================================================
# 설정
# =========================================================

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./artifacts/final_adapter")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QUESTION_MAP = {
    "accident_place": "이 사고는 어떤 도로 환경에서 발생했는가?",
    "accident_place_feature": "이 사고 장소의 특징은 무엇인가?",
    "vehicle_a_progress": "차량 A는 사고 직전 어떤 진행 상태였는가?",
    "vehicle_b_progress": "차량 B는 사고 직전 어떤 진행 상태였는가?",
    "fault_ratio": "이 사고의 과실비율은 어떻게 되는가?",
    "fault_compare": "과실비율 기준으로 더 큰 과실을 가진 차량은 누구인가?",
}

# FastAPI에서 재사용할 전역 캐시
_MODEL = None
_PROCESSOR = None


# =========================================================
# 유틸
# =========================================================

def normalize_answer(text: str) -> str:
    if not text:
        return "응답을 생성하지 못했습니다."

    text = text.strip()

    prefixes = [
        "답변:",
        "정답:",
        "출력:",
        "answer:",
        "Answer:",
        "assistant:",
        "Assistant:",
    ]
    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()

    text = re.sub(r"\s+", " ", text).strip()

    split_candidates = re.split(r"(?<=[.!?])\s+", text)
    if split_candidates and len(split_candidates[0]) > 0:
        text = split_candidates[0].strip()

    if re.fullmatch(r"\d+\s*:\s*\d+", text):
        return text.replace(" ", "")

    text = re.sub(r"이다\.?$", "입니다.", text)
    text = re.sub(r"입니다$", "입니다.", text)

    if not text.endswith((".", "!", "?")):
        text = f"{text}입니다."

    return text


def build_messages(video_path: str, question_type: str) -> list[dict]:
    if question_type not in QUESTION_MAP:
        valid_keys = ", ".join(QUESTION_MAP.keys())
        raise ValueError(
            f"지원하지 않는 question_type입니다: {question_type}\n"
            f"지원 목록: {valid_keys}"
        )

    question = QUESTION_MAP[question_type]

    system_text = (
        "당신은 교통사고 영상을 보고 질문에 대해 짧고 정확하게 답하는 AI입니다. "
        "반드시 질문에 대한 핵심 답만 간단히 출력하세요. "
        "불필요한 설명은 하지 마세요."
    )

    user_text = (
        f"question_type: {question_type}\n"
        f"question: {question}\n\n"
        "정답만 짧게 출력하세요."
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                },
                {
                    "type": "text",
                    "text": user_text,
                },
            ],
        },
    ]
    return messages


def _extract_generated_text(
    processor,
    generated_ids: torch.Tensor,
    input_ids: torch.Tensor,
) -> str:
    prompt_len = input_ids.shape[1]
    new_tokens = generated_ids[:, prompt_len:]
    decoded = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return decoded[0].strip() if decoded else ""


# =========================================================
# qwen-vl-utils fallback 처리
# =========================================================

def process_vision_inputs(messages: list[dict]) -> tuple[Optional[list], Optional[list]]:
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore

        image_inputs, video_inputs = process_vision_info(messages)
        return image_inputs, video_inputs
    except Exception:
        return None, None


# =========================================================
# 모델 로드
# =========================================================

def load_inference_objects(
    base_model_id: str = BASE_MODEL_ID,
    adapter_path: str = ADAPTER_PATH,
):
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"ADAPTER_PATH를 찾을 수 없습니다: {adapter_path}")

    print(f"[INFO] 베이스 모델 로드 중: {base_model_id}")
    base_model, _ = load_model(base_model_id)

    print(f"[INFO] 어댑터 로드 중: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        local_files_only=True,
    )

    print("[INFO] processor 로드 중...")
    processor = AutoProcessor.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        local_files_only=True,
    )

    model.eval()

    if DEVICE != "cuda":
        model = model.to(DEVICE)

    print("[INFO] 모델 준비 완료")
    return model, processor


def get_inference_objects():
    global _MODEL, _PROCESSOR

    if _MODEL is None or _PROCESSOR is None:
        _MODEL, _PROCESSOR = load_inference_objects(
            base_model_id=BASE_MODEL_ID,
            adapter_path=ADAPTER_PATH,
        )

    return _MODEL, _PROCESSOR


# =========================================================
# 추론
# =========================================================

@torch.no_grad()
def predict_one(
    model,
    processor,
    video_path: str,
    question_type: str,
    max_new_tokens: int = 32,
) -> str:
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")

    messages = build_messages(video_path=video_path, question_type=question_type)

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_inputs(messages)

    if image_inputs is None and video_inputs is None:
        inputs = processor(
            text=[prompt_text],
            return_tensors="pt",
            padding=True,
        )
    else:
        inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )

    target_device = model.device if hasattr(model, "device") else DEVICE
    inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    decoded = _extract_generated_text(
        processor=processor,
        generated_ids=generated_ids,
        input_ids=inputs["input_ids"],
    )

    return normalize_answer(decoded)


def predict_from_video(
    video_path: str,
    question_type: str,
    max_new_tokens: int = 32,
) -> str:
    model, processor = get_inference_objects()
    return predict_one(
        model=model,
        processor=processor,
        video_path=video_path,
        question_type=question_type,
        max_new_tokens=max_new_tokens,
    )