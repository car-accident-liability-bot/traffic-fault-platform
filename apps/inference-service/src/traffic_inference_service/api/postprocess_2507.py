from __future__ import annotations

import os
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from traffic_inference_service.api.prompt_config import (
    POSTPROCESS_PROMPT_MAP,
    POSTPROCESS_SYSTEM_PROMPT,
    QUESTION_MAP,
)

SUPPORTED_QUESTION_TYPES = list(QUESTION_MAP.keys())

POSTPROCESS_MODEL_ID = os.getenv("POSTPROCESS_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
POSTPROCESS_MAX_NEW_TOKENS = int(os.getenv("POSTPROCESS_MAX_NEW_TOKENS", "128"))

_2507_TOKENIZER = None
_2507_MODEL = None


def load_llm(model_path: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    후처리용 LLM 로드
    - GPU가 있으면 float16 + cuda
    - 없으면 float32 + cpu
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        print("[INFO] 2507 후처리 모델: CUDA 사용")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to("cuda")
    else:
        print("[INFO] 2507 후처리 모델: CPU 사용")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to("cpu")

    model.eval()
    return tokenizer, model


def get_2507_objects(
    model_path: str = POSTPROCESS_MODEL_ID,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    global _2507_TOKENIZER, _2507_MODEL

    if _2507_TOKENIZER is None or _2507_MODEL is None:
        _2507_TOKENIZER, _2507_MODEL = load_llm(model_path)

    return _2507_TOKENIZER, _2507_MODEL


def get_label_name(question_type: str) -> str:
    label_map = {
        "accident_place": "사고 장소",
        "accident_place_feature": "사고 장소 특징",
        "vehicle_a_progress": "차량 A 주행 상태",
        "vehicle_b_progress": "차량 B 주행 상태",
        "fault_ratio": "과실비율",
        "fault_compare": "더 큰 과실 차량",
    }

    if question_type not in label_map:
        raise ValueError(f"지원하지 않는 question_type: {question_type}")

    return label_map[question_type]


def build_prompt(question_type: str, raw_output: str) -> str:
    """
    question_type별 단일 답변 후처리 프롬프트 생성
    """
    if question_type not in POSTPROCESS_PROMPT_MAP:
        raise ValueError(f"지원하지 않는 question_type: {question_type}")

    return POSTPROCESS_PROMPT_MAP[question_type].format(
        raw_output=raw_output.strip()
    )


def generate_text(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 128,
) -> str:
    """
    LLM 생성
    """
    messages = [
        {
            "role": "system",
            "content": POSTPROCESS_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    try:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return result


def postprocess_with_2507(
    tokenizer,
    model,
    question_type: str,
    raw_output: str,
    max_new_tokens: int = 128,
) -> str:
    prompt = build_prompt(question_type, raw_output)
    final_text = generate_text(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    return final_text.strip()


def postprocess_answer(
    question_type: str,
    raw_output: str,
    model_path: str = POSTPROCESS_MODEL_ID,
    max_new_tokens: int = POSTPROCESS_MAX_NEW_TOKENS,
) -> str:
    """
    FastAPI / 추론 모듈에서 바로 호출할 후처리 함수
    """
    if question_type not in SUPPORTED_QUESTION_TYPES:
        raise ValueError(f"지원하지 않는 question_type: {question_type}")

    tokenizer, model = get_2507_objects(model_path=model_path)

    final_text = postprocess_with_2507(
        tokenizer=tokenizer,
        model=model,
        question_type=question_type,
        raw_output=raw_output,
        max_new_tokens=max_new_tokens,
    )

    return final_text