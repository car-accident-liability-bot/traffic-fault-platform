import json
import torch
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# 1. 모델 로드
# =========================
def load_llm(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()
    return tokenizer, model


# =========================
# 2. 프롬프트 생성
# =========================
def build_prompt(raw_result: Dict[str, Any]) -> str:
    prompt = f"""
너는 교통사고 분석 결과를 사용자에게 자연스럽고 일관된 한국어로 설명하는 도우미다.

다음은 한 사고 영상에 대한 예측 결과이다.

- 사고 장소: {raw_result.get("accident_place", "")}
- 사고 장소 특징: {raw_result.get("accident_place_feature", "")}
- 차량 A 주행 상태: {raw_result.get("vehicle_a_progress", "")}
- 차량 B 주행 상태: {raw_result.get("vehicle_b_progress", "")}
- 과실비율: {raw_result.get("fault_ratio", "")}
- 더 큰 과실 차량: {raw_result.get("fault_compare", "")}

요구사항:
1. 위 정보만 사용해서 2~4문장으로 설명하라.
2. 없는 사실을 절대 추가하지 마라.
3. 마지막 문장에는 과실비율과 더 큰 과실 차량을 반드시 포함하라.
4. 한국어 자연스러운 문단으로 작성하라.
5. 목록 형식 금지.

설명문만 출력하라.
"""
    return prompt.strip()


# =========================
# 3. 생성 함수
# =========================
def generate(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 256
) -> str:
    messages = [
        {"role": "system", "content": "너는 교통사고 분석 결과를 한국어로 설명하는 도우미다."},
        {"role": "user", "content": prompt}
    ]

    try:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return result


# =========================
# 4. 실행
# =========================
def main():
    # 🔥 여기에 2507 모델 경로 넣기
    model_path = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer, model = load_llm(model_path)

    # 👉 더미 결과 (best.pt 대신)
    raw_result = {
        "accident_place": "직선 도로",
        "accident_place_feature": "추돌 사고",
        "vehicle_a_progress": "선행자동차(1차사고차량)를 추돌",
        "vehicle_b_progress": "선행 자동차(1차사고차량)",
        "fault_ratio": "80:20",
        "fault_compare": "차량 A"
    }

    prompt = build_prompt(raw_result)

    print("\n[DEBUG] Prompt ====================")
    print(prompt)

    final_text = generate(tokenizer, model, prompt)

    result = {
        "raw_result": raw_result,
        "final_text": final_text
    }

    print("\n[RESULT] ====================")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()