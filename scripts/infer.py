import argparse
import json
import os
from typing import Any, Dict


def load_prompt_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic fault inference scaffold")

    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/prompt_config.json",
        help="프롬프트 config 경로"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="베이스모델 또는 튜닝모델 경로"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="입력 영상 경로"
    )
    parser.add_argument(
        "--question_type",
        type=str,
        required=True,
        choices=[
            "accident_place",
            "accident_place_feature",
            "vehicle_a_progress",
            "vehicle_b_progress",
            "fault_ratio",
            "fault_compare"
        ],
        help="드롭다운에서 선택된 질문 타입"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./artifacts/infer_result.json",
        help="결과 저장 경로"
    )

    return parser.parse_args()


def build_prompt(config: Dict[str, Any], question_type: str) -> Dict[str, str]:
    system_prompt = config["system_prompt"]
    question_templates = config["question_templates"]

    if question_type not in question_templates:
        raise ValueError(f"지원하지 않는 question_type: {question_type}")

    question = question_templates[question_type]

    return {
        "system_prompt": system_prompt,
        "question": question
    }


def dummy_model_predict(
    model_path: str,
    video_path: str,
    question_type: str,
    system_prompt: str,
    question: str
) -> str:
    """
    지금은 모델 대신 더미 응답.
    나중에 이 부분만 실제 모델 추론으로 교체하면 됨.
    """
    dummy_answers = {
        "accident_place": "직선 도로",
        "accident_place_feature": "차로변경(진로변경)",
        "vehicle_a_progress": "직진",
        "vehicle_b_progress": "직진",
        "fault_ratio": "70:30",
        "fault_compare": "차량 A"
    }

    return dummy_answers.get(question_type, "임시 응답")


def postprocess_answer(question_type: str, raw_output: str, config: Dict[str, Any]) -> str:
    output_constraints = config.get("output_constraints", {})

    if question_type == "fault_compare":
        allowed = output_constraints.get("fault_compare", ["차량 A", "차량 B", "동일"])

        if "차량 A" in raw_output or raw_output.strip() == "A":
            return "차량 A"
        if "차량 B" in raw_output or raw_output.strip() == "B":
            return "차량 B"
        if "동일" in raw_output:
            return "동일"

        # 규칙에 안 맞으면 원문 반환
        return raw_output.strip()

    if question_type == "fault_ratio":
        return raw_output.strip().replace(" ", "")

    return raw_output.strip()


def save_result(save_path: str, result: Dict[str, Any]):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    if not os.path.isfile(args.config_path):
        raise FileNotFoundError(f"config 파일이 없습니다: {args.config_path}")

    config = load_prompt_config(args.config_path)
    prompt_data = build_prompt(config, args.question_type)

    system_prompt = prompt_data["system_prompt"]
    question = prompt_data["question"]

    raw_output = dummy_model_predict(
        model_path=args.model_path,
        video_path=args.video_path,
        question_type=args.question_type,
        system_prompt=system_prompt,
        question=question
    )

    answer = postprocess_answer(args.question_type, raw_output, config)

    result = {
        "model_path": args.model_path,
        "video_path": args.video_path,
        "question_type": args.question_type,
        "system_prompt": system_prompt,
        "question": question,
        "raw_output": raw_output,
        "answer": answer
    }

    print("\n[RESULT]")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    save_result(args.save_path, result)
    print(f"\n[INFO] 저장 완료: {args.save_path}")


if __name__ == "__main__":
    main()