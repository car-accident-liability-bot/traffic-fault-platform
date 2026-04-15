import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoProcessor

# =========================================================
# 기본 질문 템플릿
# =========================================================
QUESTION_TEMPLATES = {
    "accident_place": "해당 사고는 어떤 도로 환경에서 발생했는가?",
    "accident_place_feature": "이 사고 장소의 특징은 무엇인가?",
    "vehicle_a_progress": "차량 A는 사고 직전 어떤 진행 상태였는가?",
    "vehicle_b_progress": "차량 B는 사고 직전 어떤 진행 상태였는가?",
    "fault_ratio": "이 사고의 과실비율은 어떻게 되는가?",
    "fault_compare": "과실비율 기준으로 더 큰 과실을 가진 차량은 누구인가?",
}


# =========================================================
# 유틸
# =========================================================
def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    value = v.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def resolve_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.lower()
    if dtype_str == "auto":
        if torch.cuda.is_available():
            # 보통 Ampere 이상이면 bf16 선호
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def get_device_map(device: str):
    device = device.lower()
    if device == "auto":
        return "auto" if torch.cuda.is_available() else None
    if device == "cpu":
        return None
    if device == "cuda":
        return "auto"
    raise ValueError(f"Unsupported device: {device}")


def ensure_question(question: Optional[str], question_type: Optional[str]) -> str:
    if question and question.strip():
        return question.strip()

    if question_type:
        if question_type not in QUESTION_TEMPLATES:
            raise ValueError(
                f"Unsupported question_type: {question_type}. "
                f"Available: {list(QUESTION_TEMPLATES.keys())}"
            )
        return QUESTION_TEMPLATES[question_type]

    raise ValueError("Either --question or --question_type must be provided.")


# =========================================================
# 모델 로더
# =========================================================
def load_model_class(model_type: str):
    """
    모델 config의 model_type에 맞춰 적절한 클래스를 선택한다.
    """
    if model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration

    if model_type == "qwen3_vl":
        # transformers 버전에 따라 없을 수 있음
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration

    # 일반 텍스트 모델 fallback
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM


def load_model_and_processor(
    model_path: str,
    dtype_str: str = "auto",
    device: str = "auto",
):
    model_path = str(model_path)
    dtype = resolve_dtype(dtype_str)
    device_map = get_device_map(device)

    print(f"[INFO] Loading config from: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)
    print(f"[INFO] Detected model_type: {model_type}")

    model_cls = load_model_class(model_type)

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }

    if device_map is not None:
        model_kwargs["device_map"] = device_map

    print(f"[INFO] Loading model with dtype={dtype}, device_map={device_map}")
    model = model_cls.from_pretrained(
        model_path,
        **model_kwargs,
    )
    model.eval()

    # CPU일 경우 직접 이동
    if device.lower() == "cpu":
        model.to("cpu")

    return model, processor, model_type


# =========================================================
# 멀티모달 입력 준비
# =========================================================
def build_messages(
    question: str,
    system_prompt: str,
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Qwen-VL chat template 형식으로 메시지 구성
    """
    content: List[Dict[str, Any]] = []

    if image_path:
        content.append({
            "type": "image",
            "image": image_path,
        })

    if video_path:
        content.append({
            "type": "video",
            "video": video_path,
        })

    content.append({
        "type": "text",
        "text": question,
    })

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": content,
        },
    ]
    return messages


def prepare_inputs_for_qwen_vl(
    processor,
    messages: List[Dict[str, Any]],
):
    """
    qwen_vl_utils가 있으면 이미지/영상 입력까지 처리하고,
    없으면 텍스트만 처리한다.
    """
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    has_media = False
    for msg in messages:
        for item in msg.get("content", []):
            if item.get("type") in {"image", "video"}:
                has_media = True
                break

    if has_media:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise ImportError(
                "이미지/영상 추론을 하려면 qwen_vl_utils가 필요합니다.\n"
                "설치 예시:\n"
                "  pip install qwen-vl-utils decord\n"
                "또는 텍스트만 테스트하려면 --image_path / --video_path 없이 실행하세요."
            ) from e

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[prompt_text],
            padding=True,
            return_tensors="pt",
        )

    return inputs, prompt_text


# =========================================================
# 생성
# =========================================================
def move_inputs_to_model_device(inputs, model):
    # accelerate/device_map=auto일 때 첫 파라미터 device로 보내는 방식
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    moved = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            moved[k] = v.to(model_device)
        else:
            moved[k] = v
    return moved


@torch.inference_mode()
def generate_answer(
    model,
    processor,
    model_type: str,
    messages: List[Dict[str, Any]],
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = False,
) -> Tuple[str, str]:
    inputs, prompt_text = prepare_inputs_for_qwen_vl(processor, messages)
    inputs = move_inputs_to_model_device(inputs, model)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }

    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    output_ids = model.generate(**inputs, **generate_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return output_text.strip(), prompt_text


# =========================================================
# 결과 저장
# =========================================================
def save_result(save_path: str, result: Dict[str, Any]):
    save_path = str(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved result to: {save_path}")


# =========================================================
# 메인
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Traffic fault inference script")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="베이스모델 또는 튜닝모델 경로",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="직접 질문 입력",
    )
    parser.add_argument(
        "--question_type",
        type=str,
        default=None,
        choices=list(QUESTION_TEMPLATES.keys()),
        help="미리 정의된 질문 타입 사용",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="입력 이미지 경로",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="입력 영상 경로",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "당신은 교통사고 영상 분석 보조 AI다. "
            "주어진 질문에 대해 한국어로 간결하고 직접적으로 답하라. "
            "추측이 필요한 경우 '추정'임을 짧게 표시하라. "
            "답만 짧게 말하고, 장황한 설명은 하지 마라."
        ),
        help="시스템 프롬프트",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="최대 생성 토큰 수",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="샘플링 temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="샘플링 top_p",
    )
    parser.add_argument(
        "--do_sample",
        type=str2bool,
        default=False,
        help="샘플링 사용 여부",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="모델 dtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="실행 디바이스",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="결과 JSON 저장 경로",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    question = ensure_question(args.question, args.question_type)

    if args.image_path and not Path(args.image_path).exists():
        raise FileNotFoundError(f"image_path not found: {args.image_path}")

    if args.video_path and not Path(args.video_path).exists():
        raise FileNotFoundError(f"video_path not found: {args.video_path}")

    if not args.image_path and not args.video_path:
        print("[WARN] image/video 없이 텍스트만 추론합니다.")

    model, processor, model_type = load_model_and_processor(
        model_path=args.model_path,
        dtype_str=args.dtype,
        device=args.device,
    )

    messages = build_messages(
        question=question,
        system_prompt=args.system_prompt,
        image_path=args.image_path,
        video_path=args.video_path,
    )

    answer, prompt_text = generate_answer(
        model=model,
        processor=processor,
        model_type=model_type,
        messages=messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    result = {
        "model_path": args.model_path,
        "model_type": model_type,
        "question_type": args.question_type,
        "question": question,
        "image_path": args.image_path,
        "video_path": args.video_path,
        "answer": answer,
        "prompt_text": prompt_text,
    }

    print("\n" + "=" * 80)
    print("[RESULT]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 80)

    if args.save_path:
        save_result(args.save_path, result)


if __name__ == "__main__":
    main()