"""
추론 + 평가 실행기

학습된 모델(또는 베이스라인 모델)을 사용해 test split에 대해 생성(generate)을 수행하고
metrics.py로 지표를 산출합니다.

사용 예:
    config = EvaluatorConfig(max_new_tokens=128, batch_size=1)
    summary, detail = run_evaluation(model, processor, test_samples, config)
    print(format_summary(summary, "A-1"))
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from training_runner.evaluation.metrics import (
    _BERT_SCORE_TYPES,
    aggregate_metrics,
    compute_bert_scores,
    compute_sample_metrics,
    format_summary,
)

if TYPE_CHECKING:
    from training_runner.dataset import VideoQASample

try:
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    import sys
    raise ImportError(
        f"qwen_vl_utils를 찾을 수 없습니다.\n현재 Python: {sys.executable}"
    ) from e


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class EvaluatorConfig:
    """추론·평가 설정."""
    max_new_tokens: int = 128      # 생성 최대 토큰 수
    fps: float = 1.0               # 비디오 프레임 추출률
    max_pixels: int = 360 * 420    # 비디오 프레임 최대 픽셀
    max_seq_len: int = 512         # 프롬프트 최대 토큰 길이
    system_prompt: str = (
        "당신은 교통사고 영상을 분석하는 전문 분석가입니다. "
        "제공된 블랙박스 영상을 주의 깊게 관찰하고, "
        "사고 상황에 대한 질문에 정확하고 간결하게 답변하세요."
    )
    device: str = field(default_factory=_default_device)
    verbose: bool = True           # 진행 상황 출력 여부
    # BERTScore 설정
    compute_bertscore: bool = True          # BERTScore 계산 여부
    bertscore_lang: str = "ko"              # BERTScore 언어 코드
    bertscore_model_type: str | None = None # None이면 lang으로 자동 선택
    bertscore_batch_size: int = 64          # BERTScore 배치 크기


def run_evaluation(
    model,
    processor,
    test_samples: list[VideoQASample],
    config: EvaluatorConfig | None = None,
    *,
    experiment_name: str = "",
) -> tuple[dict, list[dict]]:
    """test_samples에 대해 추론 후 평가 지표를 반환합니다.

    Args:
        model:          학습된(또는 베이스라인) 모델
        processor:      Qwen3-VL 프로세서
        test_samples:   VideoQASample 리스트 (dataset.py의 _all_samples / test split)
        config:         EvaluatorConfig (None이면 기본값 사용)
        experiment_name: 출력 헤더에 표시할 실험 이름

    Returns:
        (aggregated_metrics, per_sample_results)
        - aggregated_metrics: aggregate_metrics() 반환값
        - per_sample_results: 샘플별 dict 리스트 (prediction, answer, metrics 포함)
    """
    if config is None:
        config = EvaluatorConfig()

    model.eval()
    results: list[dict] = []
    total = len(test_samples)

    if config.verbose:
        print(f"\n[Evaluator] '{experiment_name}' 평가 시작 — {total}개 샘플")

    start = time.time()

    for idx, sample in enumerate(test_samples, 1):
        prediction = _generate_answer(model, processor, sample, config)
        sample_metrics = compute_sample_metrics(
            prediction, sample.answer, sample.question_type
        )
        results.append({
            "video_id": sample.video_id,
            "category": sample.category,
            "question_type": sample.question_type,
            "question": sample.question,
            "answer": sample.answer,
            "prediction": prediction,
            **sample_metrics,
        })

        if config.verbose and idx % 10 == 0:
            elapsed = time.time() - start
            eta = elapsed / idx * (total - idx)
            print(
                f"  [{idx}/{total}] {sample.question_type:<30} "
                f"score={sample_metrics['primary_score']:.4f}  "
                f"ETA {eta:.0f}s"
            )

    # ── BERTScore 배치 계산 ──────────────────────────────────────
    # 추론 완료 후 한 번에 계산해 GPU 메모리 스파이크를 최소화합니다.
    if config.compute_bertscore:
        bert_indices = [
            i for i, r in enumerate(results)
            if r["question_type"] in _BERT_SCORE_TYPES
        ]
        if bert_indices:
            if config.verbose:
                print(
                    f"\n[Evaluator] BERTScore 계산 중 "
                    f"({len(bert_indices)}개 샘플, lang={config.bertscore_lang}) ..."
                )
            preds = [results[i]["prediction"] for i in bert_indices]
            refs  = [results[i]["answer"]     for i in bert_indices]
            try:
                bs_scores = compute_bert_scores(
                    preds, refs,
                    lang=config.bertscore_lang,
                    model_type=config.bertscore_model_type,
                    device=config.device,
                    batch_size=config.bertscore_batch_size,
                    verbose=config.verbose,
                )
                for i, score in zip(bert_indices, bs_scores):
                    results[i]["bert_score"] = score
                if config.verbose:
                    print(f"  BERTScore 완료  평균 F1={sum(bs_scores)/len(bs_scores):.4f}")
            except ImportError as exc:
                if config.verbose:
                    print(f"  [WARNING] BERTScore 건너뜀: {exc}")
        elif config.verbose:
            print("\n[Evaluator] BERTScore 대상 샘플 없음 — 건너뜀")

    aggregated = aggregate_metrics(results)

    if config.verbose:
        print(format_summary(aggregated, experiment_name))

    return aggregated, results


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _generate_answer(
    model,
    processor,
    sample: VideoQASample,
    config: EvaluatorConfig,
) -> str:
    """단일 샘플에 대해 모델 추론 후 생성된 답변 텍스트를 반환합니다."""
    messages = [
        {"role": "system", "content": config.system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(sample.video_path),
                    "fps": config.fps,
                    "max_pixels": config.max_pixels,
                },
                {"type": "text", "text": sample.question},
            ],
        },
    ]

    # 프롬프트 텍스트 (답변 생성 시작 토큰 포함)
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_seq_len,
    )
    inputs = {k: v.to(config.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,   # 평가 시 greedy decoding
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # 입력 프롬프트 부분을 제거하고 생성된 토큰만 디코딩
    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0][input_len:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# 그래디언트 흐름 진단 유틸 (Section 4-2 대응)
# ---------------------------------------------------------------------------

def check_gradient_flow(model) -> dict[str, float]:
    """LoRA 레이어별 gradient norm을 반환합니다.

    학습 중 주기적으로 호출해 그래디언트 흐름을 진단합니다.

    Returns:
        {layer_name: grad_norm, ...}
        정상 범위: 1e-4 ~ 1e-1
    """
    grad_norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    return grad_norms


def diagnose_gradient_flow(model) -> None:
    """gradient norm을 출력하고 이상 여부를 진단합니다."""
    grad_norms = check_gradient_flow(model)

    if not grad_norms:
        print(
            "[Gradient 진단] requires_grad 파라미터에 grad가 없습니다.\n"
            "  → enable_input_require_grads() 호출 순서 확인 (get_peft_model 이전에 호출해야 함)"
        )
        return

    zero_layers = [n for n, v in grad_norms.items() if v < 1e-9]
    vision_zero = [n for n in zero_layers if "visual" in n or "vision" in n]
    lora_zero   = [n for n in zero_layers if "lora" in n.lower()]

    print(f"[Gradient 진단] 학습 파라미터 수: {len(grad_norms)}")
    print(f"  grad≈0 레이어: {len(zero_layers)} 개")

    if lora_zero:
        print(f"  ⚠ LoRA 레이어 grad=0: {len(lora_zero)}개 → target_modules 설정 확인")
    if vision_zero:
        print(f"  ⚠ Vision 레이어 grad=0: {len(vision_zero)}개 → visual encoder frozen 여부 확인")

    normal = {n: v for n, v in grad_norms.items() if 1e-4 <= v <= 1e-1}
    print(f"  정상 범위(1e-4~1e-1): {len(normal)}/{len(grad_norms)} 레이어")

    # 상위 5개 norm 값 출력
    top5 = sorted(grad_norms.items(), key=lambda x: -x[1])[:5]
    print("  [Top-5 grad norm]")
    for name, norm in top5:
        print(f"    {norm:.2e}  {name}")
