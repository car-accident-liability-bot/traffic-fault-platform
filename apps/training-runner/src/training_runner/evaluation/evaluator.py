"""
추론 + 평가 실행기

학습된 모델(또는 베이스라인 모델)을 사용해 test split에 대해 생성(generate)을 수행하고
metrics.py로 지표를 산출합니다.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
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


def _stable_path_hash(path: Path) -> str:
    """[수정] dataset과 동일하게 경로 해시를 cache key에 포함해 stem 충돌을 피합니다."""
    resolved = path.resolve(strict=False)
    return hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]


def _resolve_runtime_device(model, requested_device: str) -> str:
    """[수정] model/input device mismatch를 방지하기 위해 실제 평가 디바이스를 1회 확정합니다."""
    try:
        current_device = str(next(model.parameters()).device)
    except (StopIteration, AttributeError):
        current_device = "cpu"

    if current_device != requested_device and requested_device != "cpu":
        try:
            model.to(requested_device)
            return requested_device
        except Exception:
            return current_device
    return current_device if current_device != "meta" else requested_device


@dataclass
class EvaluatorConfig:
    """추론·평가 설정."""

    max_new_tokens: int = 64
    fps: float = 0.5
    max_pixels: int = 320 * 320
    max_seq_len: int = 512
    system_prompt: str = (
        "당신은 교통사고 영상을 분석하는 전문 분석가입니다. "
        "제공된 블랙박스 영상을 주의 깊게 관찰하고, "
        "사고 상황에 대한 질문에 정확하고 간결하게 답변하세요."
    )
    device: str = field(default_factory=_default_device)
    verbose: bool = True

    # [수정] dataset와 같은 프레임 캐시를 재사용할 수 있게 evaluator에도 cache 설정을 추가합니다.
    video_cache_dir: str = ""
    reuse_video_cache: bool = True
    sort_by_video: bool = True

    # BERTScore 설정
    compute_bertscore: bool = False
    bertscore_lang: str = "ko"
    bertscore_model_type: str | None = None
    bertscore_batch_size: int = 32


def run_evaluation(
    model,
    processor,
    test_samples: list[VideoQASample],
    config: EvaluatorConfig | None = None,
    *,
    experiment_name: str = "",
) -> tuple[dict, list[dict]]:
    """test_samples에 대해 추론 후 평가 지표를 반환합니다."""
    if config is None:
        config = EvaluatorConfig()

    # [수정] 실제 모델이 올라간 장치에 맞춰 입력 텐서를 보내 device mismatch를 막습니다.
    runtime_device = _resolve_runtime_device(model, config.device)
    model.eval()
    results: list[dict] = []
    total = len(test_samples)

    # [수정] 같은 영상을 묶어 순회하면 cache hit율이 올라 최종 평가 시간이 줄어듭니다.
    if config.sort_by_video:
        ordered_samples = sorted(
            test_samples,
            key=lambda sample: (str(sample.video_path), sample.question_type, sample.video_id),
        )
    else:
        ordered_samples = test_samples

    # [수정] evaluator 레벨 메모리 cache.
    video_cache: dict[str, list] = {}
    # [수정] 같은 비디오의 비전 텐서를 재사용해 평가 전처리 병목을 줄입니다.
    visual_feature_cache: dict[str, dict[str, torch.Tensor]] = {}
    # [수정] text-only input_ids와 multimodal input_ids가 실제로 같은지 1회만 검증합니다.
    text_merge_state: dict[str, bool | None] = {"supported": None}

    if config.verbose:
        print(f"\n[Evaluator] '{experiment_name}' 평가 시작 — {total}개 샘플")

    start = time.time()

    for idx, sample in enumerate(ordered_samples, 1):
        prediction = _generate_answer(
            model,
            processor,
            sample,
            config,
            video_cache=video_cache,
            visual_feature_cache=visual_feature_cache,
            runtime_device=runtime_device,
            text_merge_state=text_merge_state,
        )
        sample_metrics = compute_sample_metrics(
            prediction,
            sample.answer,
            sample.question_type,
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
                f"score={sample_metrics['primary_score']:.4f}  ETA {eta:.0f}s"
            )

    if config.compute_bertscore:
        bert_indices = [
            i for i, result in enumerate(results)
            if result["question_type"] in _BERT_SCORE_TYPES
        ]
        if bert_indices:
            if config.verbose:
                print(
                    f"\n[Evaluator] BERTScore 계산 중 "
                    f"({len(bert_indices)}개 샘플, lang={config.bertscore_lang}) ..."
                )
            preds = [results[i]["prediction"] for i in bert_indices]
            refs = [results[i]["answer"] for i in bert_indices]
            try:
                bs_scores = compute_bert_scores(
                    preds,
                    refs,
                    lang=config.bertscore_lang,
                    model_type=config.bertscore_model_type,
                    device=runtime_device,
                    batch_size=config.bertscore_batch_size,
                    verbose=config.verbose,
                )
                for i, score in zip(bert_indices, bs_scores):
                    results[i]["bert_score"] = score
                if config.verbose:
                    print(f"  BERTScore 완료  평균 F1={sum(bs_scores) / len(bs_scores):.4f}")
            except ImportError as exc:
                if config.verbose:
                    print(f"  [WARNING] BERTScore 건너뜀: {exc}")
        elif config.verbose:
            print("\n[Evaluator] BERTScore 대상 샘플 없음 — 건너뜀")

    aggregated = aggregate_metrics(results)

    if config.verbose:
        print(format_summary(aggregated, experiment_name))

    return aggregated, results


def _build_video_cache_key(video_path: Path, fps: float, max_pixels: int) -> str:
    return f"{video_path.stem}_{_stable_path_hash(video_path)}_fps{fps}_px{max_pixels}"


def _load_or_decode_video_inputs(
    sample: VideoQASample,
    config: EvaluatorConfig,
    *,
    video_cache: dict[str, list],
) -> list:
    cache_key = f"{sample.video_path}|fps={config.fps}|px={config.max_pixels}"
    if cache_key in video_cache:
        return video_cache[cache_key]

    cache_dir = Path(config.video_cache_dir) if config.video_cache_dir else None
    cache_file: Path | None = None

    if config.reuse_video_cache and cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        disk_key = _build_video_cache_key(Path(sample.video_path), config.fps, config.max_pixels)
        cache_file = cache_dir / f"{disk_key}.npy"
        if cache_file.exists():
            try:
                frames = np.load(str(cache_file), allow_pickle=False)
                video_inputs = [frames]
                video_cache[cache_key] = video_inputs
                return video_inputs
            except (ValueError, OSError):
                cache_file.unlink(missing_ok=True)

    _, video_inputs = process_vision_info([
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(sample.video_path),
                    "fps": config.fps,
                    "max_pixels": config.max_pixels,
                }
            ],
        }
    ])

    # [수정] 평가에서 새로 디코딩한 결과도 캐시 파일로 남겨 다음 실행에 재사용합니다.
    if config.reuse_video_cache and cache_file is not None and video_inputs:
        tmp_file = cache_file.with_name(f"{cache_file.stem}.{os.getpid()}.tmp.npy")
        np.save(str(tmp_file), video_inputs[0])
        os.replace(str(tmp_file), str(cache_file))

    video_cache[cache_key] = video_inputs
    return video_inputs


def _build_text_inputs(processor, prompt_text: str, config: EvaluatorConfig) -> dict[str, torch.Tensor]:
    """[수정] 비디오와 무관한 input_ids/attention_mask는 text-only tokenization으로 생성합니다."""
    return processor(
        text=[prompt_text],
        images=None,
        videos=None,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_seq_len,
    )



def _extract_visual_features(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """[수정] 질문과 무관한 비전 텐서만 분리해 비디오 단위로 재사용합니다."""
    visual: dict[str, torch.Tensor] = {}
    for key in ["pixel_values", "image_grid_thw", "video_grid_thw"]:
        if key in batch:
            visual[key] = batch[key].clone()
    return visual



def _build_generation_inputs(
    processor,
    sample: VideoQASample,
    config: EvaluatorConfig,
    *,
    video_cache: dict[str, list],
    visual_feature_cache: dict[str, dict[str, torch.Tensor]],
    text_merge_state: dict[str, bool | None],
) -> dict[str, torch.Tensor]:
    """[수정] text-only + cached visual merge가 가능하면 같은 비디오의 비전 전처리를 한 번만 사용합니다."""
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
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    text_inputs = _build_text_inputs(processor, prompt_text, config)
    visual_cache_key = str(sample.video_path)
    cached_visual = visual_feature_cache.get(visual_cache_key)
    if cached_visual is not None and text_merge_state["supported"] is True:
        merged_inputs = dict(text_inputs)
        merged_inputs.update(cached_visual)
        return merged_inputs

    video_inputs = _load_or_decode_video_inputs(
        sample,
        config,
        video_cache=video_cache,
    )
    multimodal_inputs = processor(
        text=[prompt_text],
        images=None,
        videos=video_inputs,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_seq_len,
    )

    ids_match = torch.equal(multimodal_inputs["input_ids"], text_inputs["input_ids"])
    mask_match = torch.equal(multimodal_inputs["attention_mask"], text_inputs["attention_mask"])
    if text_merge_state["supported"] is None:
        text_merge_state["supported"] = bool(ids_match and mask_match)
        if config.verbose:
            print(
                "[Evaluator] text-only + visual cache 병합 "
                f"{'활성화' if text_merge_state['supported'] else '비활성화'}"
            )

    if text_merge_state["supported"] and ids_match and mask_match:
        visual_feature_cache[visual_cache_key] = _extract_visual_features(multimodal_inputs)
        merged_inputs = dict(text_inputs)
        merged_inputs.update(visual_feature_cache[visual_cache_key])
        return merged_inputs

    # [수정] 호환되지 않으면 안전하게 기존 multimodal processor 결과를 그대로 사용합니다.
    text_merge_state["supported"] = False
    return multimodal_inputs



def _generate_answer(
    model,
    processor,
    sample: VideoQASample,
    config: EvaluatorConfig,
    *,
    video_cache: dict[str, list],
    visual_feature_cache: dict[str, dict[str, torch.Tensor]],
    runtime_device: str,
    text_merge_state: dict[str, bool | None],
) -> str:
    """단일 샘플에 대해 모델 추론 후 생성된 답변 텍스트를 반환합니다."""
    inputs = _build_generation_inputs(
        processor,
        sample,
        config,
        video_cache=video_cache,
        visual_feature_cache=visual_feature_cache,
        text_merge_state=text_merge_state,
    )
    inputs = {key: value.to(runtime_device) for key, value in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0][input_len:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def check_gradient_flow(model) -> dict[str, float]:
    """LoRA 레이어별 gradient norm을 반환합니다."""
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

    zero_layers = [name for name, value in grad_norms.items() if value < 1e-9]
    vision_zero = [name for name in zero_layers if "visual" in name or "vision" in name]
    lora_zero = [name for name in zero_layers if "lora" in name.lower()]

    print(f"[Gradient 진단] 학습 파라미터 수: {len(grad_norms)}")
    print(f"  grad≈0 레이어: {len(zero_layers)} 개")

    if lora_zero:
        print(f"  ⚠ LoRA 레이어 grad=0: {len(lora_zero)}개 → target_modules 설정 확인")
    if vision_zero:
        print(f"  ⚠ Vision 레이어 grad=0: {len(vision_zero)}개 → visual encoder frozen 여부 확인")

    normal = {name: value for name, value in grad_norms.items() if 1e-4 <= value <= 1e-1}
    print(f"  정상 범위(1e-4~1e-1): {len(normal)}/{len(grad_norms)} 레이어")

    top5 = sorted(grad_norms.items(), key=lambda item: -item[1])[:5]
    print("  [Top-5 grad norm]")
    for name, norm in top5:
        print(f"    {norm:.2e}  {name}")
