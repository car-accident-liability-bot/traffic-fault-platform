from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import requests
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from traffic_inference_service.api.postprocess_2507 import postprocess_answer
from traffic_inference_service.api.prompt_config import QUESTION_MAP, SYSTEM_PROMPT

# =========================================================
# 설정
# =========================================================

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")

ADAPTER_URL = os.getenv(
    "ADAPTER_URL",
    "https://data.taeo-dev.com/dataset/traffic/final_adapter/",
)

LOCAL_ADAPTER_PATH = Path(__file__).resolve().parents[4] / "artifacts" / "final_adapter"
ADAPTER_PATH = Path(os.getenv("ADAPTER_PATH", str(LOCAL_ADAPTER_PATH)))

USE_2507_POSTPROCESS = os.getenv("USE_2507_POSTPROCESS", "true").lower() == "true"

# FastAPI 전역 캐시
_MODEL = None
_PROCESSOR = None

# adapter에 최소 필요 파일
REQUIRED_ADAPTER_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
]

# 선택 다운로드 파일
OPTIONAL_ADAPTER_FILES = [
    "README.md",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "processor_config.json",
]


# =========================================================
# 디바이스 / 로그 유틸
# =========================================================

def get_runtime_device() -> tuple[str, torch.dtype]:
    """
    GPU가 가능하면 cuda + bfloat16 우선 사용.
    아니면 cpu + float32 사용.
    """
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def log_runtime_device() -> tuple[str, torch.dtype]:
    device, dtype = get_runtime_device()

    print("[INFO] ===== Runtime Device Info =====")
    print(f"[INFO] torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"[INFO] selected_device: {device}")
    print(f"[INFO] selected_dtype: {dtype}")

    if device == "cuda":
        try:
            current_idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_idx)
            props = torch.cuda.get_device_properties(current_idx)
            total_vram_gb = props.total_memory / (1024 ** 3)

            print(f"[INFO] cuda_device_index: {current_idx}")
            print(f"[INFO] cuda_device_name: {gpu_name}")
            print(f"[INFO] cuda_total_vram_gb: {total_vram_gb:.2f}")
        except Exception as e:
            print(f"[WARN] CUDA 상세 정보 조회 실패: {e}")
    else:
        print("[INFO] CUDA unavailable -> CPU fallback")

    print("[INFO] =================================")
    return device, dtype


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

    user_text = (
        f"question_type: {question_type}\n"
        f"question: {question}\n\n"
        "정답만 짧게 출력하세요."
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
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
# NAS -> local adapter 다운로드
# =========================================================

def _download_file(file_url: str, save_path: Path, timeout: int = 60) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(file_url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def ensure_adapter_files(
    adapter_dir: Path,
    adapter_url: str,
    required_files: list[str] | None = None,
    optional_files: list[str] | None = None,
) -> Path:
    required_files = required_files or REQUIRED_ADAPTER_FILES
    optional_files = optional_files or OPTIONAL_ADAPTER_FILES

    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_url = adapter_url.rstrip("/")

    missing_required = [name for name in required_files if not (adapter_dir / name).exists()]

    if not missing_required:
        print(f"[INFO] 로컬 adapter 사용: {adapter_dir}")
        return adapter_dir

    print("[INFO] 로컬 adapter 일부 누락 -> NAS에서 다운로드 시도")
    print(f"[INFO] NAS URL: {adapter_url}")
    print(f"[INFO] 대상 경로: {adapter_dir}")

    for filename in missing_required:
        file_url = f"{adapter_url}/{filename}"
        save_path = adapter_dir / filename
        print(f"[INFO] 다운로드 중: {file_url}")
        _download_file(file_url, save_path)

    for filename in optional_files:
        save_path = adapter_dir / filename
        if save_path.exists():
            continue

        file_url = f"{adapter_url}/{filename}"
        try:
            print(f"[INFO] 선택 파일 다운로드 시도: {file_url}")
            _download_file(file_url, save_path)
        except Exception as e:
            print(f"[WARN] 선택 파일 다운로드 실패: {filename} | {e}")

    still_missing = [name for name in required_files if not (adapter_dir / name).exists()]
    if still_missing:
        raise FileNotFoundError(
            f"adapter 필수 파일이 없습니다: {still_missing}\n"
            f"ADAPTER_URL={adapter_url}\n"
            f"ADAPTER_PATH={adapter_dir}"
        )

    print(f"[INFO] adapter 다운로드/검증 완료: {adapter_dir}")
    return adapter_dir


# =========================================================
# 모델 로드
# =========================================================

def load_inference_objects(
    base_model_id: str = BASE_MODEL_ID,
    adapter_path: Path = ADAPTER_PATH,
    adapter_url: str = ADAPTER_URL,
):
    device, dtype = log_runtime_device()

    adapter_dir = ensure_adapter_files(
        adapter_dir=Path(adapter_path),
        adapter_url=adapter_url,
    )

    print(f"[INFO] 베이스 모델 로드 중: {base_model_id}")
    print(f"[INFO] model_load_device_policy: prefer_gpu_if_available")
    print(f"[INFO] model_target_device: {device}")
    print(f"[INFO] model_target_dtype: {dtype}")

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"[INFO] 어댑터 로드 중: {adapter_dir}")
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        local_files_only=True,
        is_trainable=False,
    )

    if device == "cuda":
        print("[INFO] GPU 사용 가능 -> model.to('cuda') 실행")
        model = model.to("cuda")
    else:
        print("[INFO] GPU 사용 불가 -> CPU 유지")

    print("[INFO] processor 로드 중...")
    processor = AutoProcessor.from_pretrained(
        base_model_id,
        trust_remote_code=True,
    )

    model.eval()

    try:
        actual_device = next(model.parameters()).device
    except StopIteration:
        actual_device = "unknown"

    print(
        f"[INFO] 모델 준비 완료 | "
        f"base_model_id: {base_model_id} | "
        f"adapter_dir: {adapter_dir} | "
        f"requested_device: {device} | "
        f"actual_device: {actual_device} | "
        f"dtype: {dtype}"
    )

    if device == "cuda":
        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"[INFO] cuda_memory_allocated_gb: {allocated:.2f}")
            print(f"[INFO] cuda_memory_reserved_gb: {reserved:.2f}")
        except Exception as e:
            print(f"[WARN] CUDA 메모리 로그 출력 실패: {e}")

    return model, processor


def get_inference_objects():
    global _MODEL, _PROCESSOR

    if _MODEL is None or _PROCESSOR is None:
        print("[INFO] 캐시된 모델 없음 -> 최초 1회 로드 수행")
        _MODEL, _PROCESSOR = load_inference_objects(
            base_model_id=BASE_MODEL_ID,
            adapter_path=ADAPTER_PATH,
            adapter_url=ADAPTER_URL,
        )
    else:
        try:
            cached_device = next(_MODEL.parameters()).device
        except Exception:
            cached_device = "unknown"
        print(f"[INFO] 캐시된 모델 재사용 | device: {cached_device}")

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

    target_device = next(model.parameters()).device
    print(f"[INFO] 추론 device: {target_device}")

    inputs = {
        k: v.to(target_device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

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

    raw_output = normalize_answer(decoded)

    if USE_2507_POSTPROCESS:
        try:
            final_output = postprocess_answer(
                question_type=question_type,
                raw_output=raw_output,
            )
            return final_output
        except Exception as e:
            print(f"[WARN] 2507 후처리 실패, 원본 답변 반환: {e}")
            return raw_output

    return raw_output


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
