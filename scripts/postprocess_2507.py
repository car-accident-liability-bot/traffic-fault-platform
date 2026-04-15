import argparse
import json
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


SUPPORTED_QUESTION_TYPES = [
    "accident_place",
    "accident_place_feature",
    "vehicle_a_progress",
    "vehicle_b_progress",
    "fault_ratio",
    "fault_compare",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-question postprocessing with Qwen 2507"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="후처리용 LLM 모델 경로 또는 Hugging Face repo id"
    )
    parser.add_argument(
        "--question_type",
        type=str,
        required=True,
        choices=SUPPORTED_QUESTION_TYPES,
        help="드롭다운에서 선택된 질문 타입"
    )
    parser.add_argument(
        "--raw_output",
        type=str,
        required=True,
        help="메인 모델이 반환한 원본 짧은 답변"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="생성 최대 토큰 수"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="결과 저장 경로(json). 비워두면 저장하지 않음"
    )

    return parser.parse_args()


def load_llm(model_path: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    후처리용 LLM 로드
    - GPU가 있으면 float16 + cuda
    - 없으면 float32 + cpu
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    if torch.cuda.is_available():
        print("[INFO] CUDA 사용")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            trust_remote_code=True
        ).to("cuda")
    else:
        print("[INFO] CPU 사용")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            trust_remote_code=True
        ).to("cpu")

    model.eval()
    return tokenizer, model


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
    raw_output = raw_output.strip()

    prompt_map = {
        "accident_place": f"""
너는 교통사고 분석 결과를 사용자에게 자연스럽고 간결한 한국어로 설명하는 도우미다.

다음은 사고 장소 예측 결과이다.
- 사고 장소: {raw_output}

요구사항:
1. 위 정보만 사용해서 한국어 한 문장으로 설명하라.
2. 없는 사실을 추가하지 마라.
3. "판단됩니다" 또는 "보입니다" 중 자연스러운 표현을 사용하라.
4. 설명문만 출력하라.
5. 반드시 존댓말 종결형만 사용하라.
6. 문장 끝은 "~입니다.", "~됩니다.", "~보입니다.", "~판단됩니다." 형태만 사용하라.
7. "~이다", "~였다", "~로 보인다" 같은 평서형은 사용하지 마라.
""".strip(),

        "accident_place_feature": f"""
너는 교통사고 분석 결과를 사용자에게 자연스럽고 간결한 한국어로 설명하는 도우미다.

다음은 사고 장소 특징 예측 결과이다.
- 사고 장소 특징: {raw_output}

요구사항:
1. 위 정보만 사용해서 한국어 한 문장으로 설명하라.
2. 없는 사실을 추가하지 마라.
3. 설명문만 출력하라.
4. 반드시 존댓말 종결형만 사용하라.
5. 문장 끝은 "~입니다.", "~됩니다.", "~보입니다.", "~판단됩니다." 형태만 사용하라.
6. "~이다", "~였다", "~로 보인다" 같은 평서형은 사용하지 마라.
""".strip(),

        "vehicle_a_progress": f"""
너는 교통사고 분석 결과를 사용자에게 자연스럽고 간결한 한국어로 설명하는 도우미다.

다음은 차량 A 주행 상태 예측 결과이다.
- 차량 A 주행 상태: {raw_output}

요구사항:
1. 위 정보만 사용해서 한국어 한 문장으로 설명하라.
2. 차량 A와 차량 B의 관계를 임의로 해석하지 마라.
3. 없는 사실을 추가하지 마라.
4. 입력 표현을 가능한 유지하라.
5. 설명문만 출력하라.
6. 반드시 존댓말 종결형만 사용하라.
7. 문장 끝은 "~입니다.", "~됩니다.", "~보입니다.", "~판단됩니다." 형태만 사용하라.
8. "~이다", "~였다", "~로 보인다" 같은 평서형은 사용하지 마라.
""".strip(),

        "vehicle_b_progress": f"""
너는 교통사고 분석 결과를 사용자에게 자연스럽고 간결한 한국어로 설명하는 도우미다.

다음은 차량 B 주행 상태 예측 결과이다.
- 차량 B 주행 상태: {raw_output}

요구사항:
1. 위 정보만 사용해서 한국어 한 문장으로 설명하라.
2. 차량 A와 차량 B의 관계를 임의로 해석하지 마라.
3. 없는 사실을 추가하지 마라.
4. 입력 표현을 가능한 유지하라.
5. 설명문만 출력하라.
6. 반드시 존댓말 종결형만 사용하라.
7. 문장 끝은 "~입니다.", "~됩니다.", "~보입니다.", "~판단됩니다." 형태만 사용하라.
8. "~이다", "~였다", "~로 보인다" 같은 평서형은 사용하지 마라.
""".strip(),

        "fault_ratio": f"""
너는 교통사고 분석 결과를 사용자에게 자연스럽고 간결한 한국어로 설명하는 도우미다.

다음은 과실비율 예측 결과이다.
- 과실비율: {raw_output}

요구사항:
1. 위 정보만 사용해서 한국어 한 문장으로 설명하라.
2. 없는 사실을 추가하지 마라.
3. 설명문만 출력하라.
4. 반드시 존댓말 종결형만 사용하라.
5. 문장 끝은 "~입니다.", "~됩니다.", "~보입니다.", "~판단됩니다." 형태만 사용하라.
6. "~이다", "~였다", "~로 보인다" 같은 평서형은 사용하지 마라.
""".strip(),

        "fault_compare": f"""
너는 교통사고 분석 결과를 사용자에게 자연스럽고 간결한 한국어로 설명하는 도우미다.

다음은 더 큰 과실 차량 예측 결과이다.
- 더 큰 과실 차량: {raw_output}

요구사항:
1. 위 정보만 사용해서 한국어 한 문장으로 설명하라.
2. 없는 사실을 추가하지 마라.
3. 설명문만 출력하라.
4. 반드시 존댓말 종결형만 사용하라.
5. 문장 끝은 "~입니다.", "~됩니다.", "~보입니다.", "~판단됩니다." 형태만 사용하라.
6. "~이다", "~였다", "~로 보인다" 같은 평서형은 사용하지 마라.
""".strip(),
    }

    if question_type not in prompt_map:
        raise ValueError(f"지원하지 않는 question_type: {question_type}")

    return prompt_map[question_type]


def generate_text(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 128
) -> str:
    """
    LLM 생성
    """
    messages = [
        {
            "role": "system",
            "content": "너는 교통사고 분석 결과를 한국어로 정리하는 도우미다."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
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
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    return result


def postprocess_with_2507(
    tokenizer,
    model,
    question_type: str,
    raw_output: str,
    max_new_tokens: int = 128
) -> str:
    prompt = build_prompt(question_type, raw_output)
    final_text = generate_text(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens
    )
    return final_text


def save_result(save_path: str, result: dict):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    tokenizer, model = load_llm(args.model_path)

    prompt = build_prompt(
        question_type=args.question_type,
        raw_output=args.raw_output
    )

    print("\n[DEBUG] Prompt ====================")
    print(prompt)

    final_text = postprocess_with_2507(
        tokenizer=tokenizer,
        model=model,
        question_type=args.question_type,
        raw_output=args.raw_output,
        max_new_tokens=args.max_new_tokens
    )

    result = {
        "model_path": args.model_path,
        "question_type": args.question_type,
        "label_name": get_label_name(args.question_type),
        "raw_output": args.raw_output,
        "final_text": final_text
    }

    print("\n[RESULT] ====================")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.save_path.strip():
        save_result(args.save_path, result)
        print(f"\n[INFO] 저장 완료: {args.save_path}")


if __name__ == "__main__":
    main()