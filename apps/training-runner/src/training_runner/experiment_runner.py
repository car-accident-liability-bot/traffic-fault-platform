"""
실험 오케스트레이터

실험 카탈로그에서 실험을 선택하고, 학습 → 평가 → 결과 저장 파이프라인을 실행합니다.

사용 예:
    # 특정 실험 실행
    python -m training_runner.experiment_runner --exp A-1 A-2

    # Phase 전체 실행
    python -m training_runner.experiment_runner --phase 1

    # 등록된 실험 목록 확인
    python -m training_runner.experiment_runner --list

    # Phase 0 베이스라인만 실행
    python -m training_runner.experiment_runner --phase 0

    # 결과 비교 테이블 출력 (이미 완료된 실험 기준)
    python -m training_runner.experiment_runner --compare
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from training_runner.configs.experiment_config import (
    EXPERIMENT_CATALOG,
    ExperimentConfig,
    get_experiments_by_phase,
    list_experiments,
)
from training_runner.configs.training_config import TrainingConfig
from training_runner.evaluation.evaluator import EvaluatorConfig, run_evaluation
from training_runner.evaluation.metrics import format_summary

# 결과 저장 루트
DEFAULT_RESULTS_DIR = Path("outputs/experiments")
DEFAULT_CHECKPOINT_ROOT = "checkpoints"


# ---------------------------------------------------------------------------
# 메인 오케스트레이터
# ---------------------------------------------------------------------------

def run_experiment(
    exp: ExperimentConfig,
    base_config: TrainingConfig,
    *,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    checkpoint_root: str = DEFAULT_CHECKPOINT_ROOT,
    force_rerun: bool = False,
) -> dict:
    """단일 실험을 실행하고 결과를 저장합니다.

    Args:
        exp:              실험 설정
        base_config:      기준 TrainingConfig (경로, 모델 ID 등)
        results_dir:      결과 JSON 저장 루트
        checkpoint_root:  체크포인트 루트
        force_rerun:      True면 이미 완료된 결과도 재실행

    Returns:
        집계된 평가 지표 dict
    """
    exp_dir = results_dir / exp.name
    summary_path = exp_dir / "summary.json"

    if summary_path.exists() and not force_rerun:
        print(f"\n[{exp.name}] 이미 완료된 결과 발견 — 스킵 (--force-rerun으로 재실행)")
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)

    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"실험: {exp.name}  Phase {exp.phase}")
    print(f"설명: {exp.description}")
    print(f"{'='*60}")

    # TrainingConfig 생성
    training_cfg = exp.build_training_config(
        base_config=base_config,
        checkpoint_root=checkpoint_root,
    )

    start_time = time.time()

    # --- 학습 단계 ---
    if exp.skip_training:
        print(f"[{exp.name}] 학습 스킵 (Phase 0 베이스라인)")
        model, processor = _load_base_model(training_cfg)
    else:
        print(f"[{exp.name}] 학습 시작...")
        model, processor = _run_training_and_load(training_cfg)

    # --- 평가 단계 ---
    print(f"\n[{exp.name}] 평가 시작...")
    test_samples = _build_test_samples(training_cfg)

    eval_config = EvaluatorConfig(
        fps=exp.fps,
        max_pixels=exp.max_pixels,
        system_prompt=exp.system_prompt,
        verbose=True,
    )
    aggregated, per_sample = run_evaluation(
        model, processor, test_samples,
        config=eval_config,
        experiment_name=exp.name,
    )

    # --- 결과 저장 ---
    elapsed = time.time() - start_time
    result_payload = {
        "experiment": exp.name,
        "phase": exp.phase,
        "description": exp.description,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "config": {
            "lora_r": exp.lora_r,
            "lora_alpha": exp.lora_alpha,
            "lora_dropout": exp.lora_dropout,
            "fps": exp.fps,
            "max_pixels": exp.max_pixels,
            "learning_rate": exp.learning_rate,
            "num_train_epochs": exp.num_train_epochs,
            "system_prompt_key": exp.system_prompt_key,
            "max_samples_per_category": exp.max_samples_per_category,
            "skip_training": exp.skip_training,
        },
        "metrics": aggregated,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    detail_path = exp_dir / "per_sample.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(per_sample, f, ensure_ascii=False, indent=2)

    print(f"\n[{exp.name}] 결과 저장: {exp_dir}  ({elapsed:.0f}s)")

    _update_comparison_csv(results_dir, result_payload)

    # 메모리 해제
    del model
    torch.cuda.empty_cache()

    return result_payload


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _load_base_model(config: TrainingConfig):
    """베이스라인 모델(어댑터 없음)을 로드합니다."""
    from traffic_ai_core.Qwen3_VL_4B_Instruct.model.model import load_model
    print(f"  베이스 모델 로드: {config.model_id}")
    return load_model(config.model_id)


def _run_training_and_load(config: TrainingConfig):
    """학습을 실행하고 최종 어댑터가 적용된 모델을 반환합니다."""
    from peft import PeftModel
    from traffic_ai_core.Qwen3_VL_4B_Instruct.model.model import load_model
    from training_runner.train import run_training

    run_training(config)

    # 학습 완료 후 final_adapter 로드
    adapter_dir = Path(config.checkpoint_dir) / "final_adapter"
    print(f"  어댑터 로드: {adapter_dir}")
    base_model, processor = load_model(config.model_id)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    return model, processor


def _build_test_samples(config: TrainingConfig):
    """test split의 VideoQASample 리스트를 구성합니다 (계층적 분할 사용)."""
    from transformers import AutoProcessor

    from training_runner.dataset import TrafficAccidentQADataset, stratified_split_by_category

    processor = AutoProcessor.from_pretrained(config.model_id)
    full_ds = TrafficAccidentQADataset(
        qa_json_path=config.qa_json_path,
        raw_video_root=config.raw_video_root,
        label_root=config.label_root,
        processor=processor,
        fps=config.fps,
        max_pixels=config.max_pixels,
        max_seq_len=config.max_seq_len,
        question_types=config.question_types,
        max_samples_per_category=config.max_samples_per_category,
    )
    all_samples = full_ds.get_all_samples()

    # case_code별 계층적 분할 — build_dataloaders와 동일한 로직으로 test split 재현
    _, _, test_ids = stratified_split_by_category(
        all_samples,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    return [s for s in all_samples if s.video_id in test_ids]


def _update_comparison_csv(results_dir: Path, result: dict) -> None:
    """results_dir/comparison.csv에 이번 실험 결과를 추가(upsert)합니다."""
    import csv

    csv_path = results_dir / "comparison.csv"
    overall = result.get("metrics", {}).get("overall", {})
    by_qtype = result.get("metrics", {}).get("by_question_type", {})

    row = {
        "experiment": result["experiment"],
        "phase": result["phase"],
        "description": result["description"],
        "timestamp": result["timestamp"],
        "fault_ratio_em": overall.get("fault_ratio_em", ""),
        "fault_ratio_mae": overall.get("fault_ratio_mae_mean", ""),
        "fault_compare_acc": overall.get("fault_compare_acc", ""),
        "rouge_l_mean": overall.get("rouge_l_mean", ""),
        "lora_r": result["config"]["lora_r"],
        "fps": result["config"]["fps"],
        "lr": result["config"]["learning_rate"],
        "system_prompt": result["config"]["system_prompt_key"],
    }
    # question_type별 rouge_l 추가
    for qtype in [
        "accident_place", "accident_place_feature",
        "vehicle_a_progress", "vehicle_b_progress",
    ]:
        row[f"rouge_l_{qtype}"] = (
            by_qtype.get(qtype, {}).get("rouge_l_mean", "")
        )

    fieldnames = list(row.keys())

    # 기존 CSV 읽기
    existing: list[dict] = []
    if csv_path.exists():
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing = [r for r in reader if r.get("experiment") != row["experiment"]]
        # 기존 필드셋과 합집합
        if existing:
            fieldnames = list(
                dict.fromkeys(list(existing[0].keys()) + fieldnames)
            )

    existing.append(row)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)

    print(f"  비교 CSV 업데이트: {csv_path}")


# ---------------------------------------------------------------------------
# 결과 비교 출력
# ---------------------------------------------------------------------------

def print_comparison(results_dir: Path = DEFAULT_RESULTS_DIR) -> None:
    """완료된 모든 실험의 핵심 지표를 표로 출력합니다."""
    summaries = []
    for summary_path in sorted(results_dir.rglob("summary.json")):
        with open(summary_path, encoding="utf-8") as f:
            summaries.append(json.load(f))

    if not summaries:
        print("완료된 실험이 없습니다.")
        return

    summaries.sort(key=lambda x: (x["phase"], x["experiment"]))

    header = (
        f"{'실험':<25} {'Phase':<7} "
        f"{'fault_ratio EM':>14} {'ROUGE-L':>9} {'compare Acc':>12} "
        f"{'rank':>6} {'fps':>5} {'LR':>8}"
    )
    print("\n" + "=" * len(header))
    print("실험 비교 결과")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for s in summaries:
        overall = s.get("metrics", {}).get("overall", {})
        cfg = s.get("config", {})

        def _v(x):
            return f"{x:.4f}" if isinstance(x, float) else "N/A"

        print(
            f"{s['experiment']:<25} Phase {s['phase']}  "
            f"{_v(overall.get('fault_ratio_em')):>14} "
            f"{_v(overall.get('rouge_l_mean')):>9} "
            f"{_v(overall.get('fault_compare_acc')):>12} "
            f"{cfg.get('lora_r', '-'):>6} "
            f"{cfg.get('fps', '-'):>5} "
            f"{cfg.get('learning_rate', '-'):>8}"
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="교통사고 과실비율 Video QA 실험 오케스트레이터"
    )

    # 데이터 경로 (공통)
    d = TrainingConfig()
    parser.add_argument("--qa-json",     default=d.qa_json_path,    help="QA 데이터셋 JSON 경로")
    parser.add_argument("--video-dir",   default=d.raw_video_root,  help="원본 영상 루트")
    parser.add_argument("--label-dir",   default=d.label_root,      help="라벨 JSON 루트")
    parser.add_argument("--model-id",    default=d.model_id,        help="HuggingFace 모델 ID")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="결과 저장 루트")
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT, help="체크포인트 루트")

    # 실험 선택
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--exp", nargs="+", metavar="NAME",
        help="실행할 실험 이름 (예: A-1 A-2 B-1)"
    )
    group.add_argument(
        "--phase", type=int, metavar="N",
        help="실행할 Phase 번호 (0, 1, 2)"
    )
    group.add_argument("--list",    action="store_true", help="등록된 실험 목록 출력")
    group.add_argument("--compare", action="store_true", help="완료된 실험 비교 테이블 출력")

    # 제어 플래그
    parser.add_argument("--force-rerun", action="store_true",
                        help="이미 완료된 실험도 재실행")
    parser.add_argument("--smoke-test",  action="store_true",
                        help="학습 max_steps=2로 파이프라인 빠른 검증")
    parser.add_argument("--seed",        type=int, default=d.seed)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # 목록 출력
    if args.list:
        list_experiments()
        return

    # 비교 테이블 출력
    if args.compare:
        print_comparison(Path(args.results_dir))
        return

    # 실험 목록 결정
    if args.exp:
        experiments = []
        for name in args.exp:
            if name not in EXPERIMENT_CATALOG:
                raise ValueError(
                    f"실험 '{name}'을 찾을 수 없습니다. "
                    f"--list로 등록된 실험을 확인하세요."
                )
            experiments.append(EXPERIMENT_CATALOG[name])
    elif args.phase is not None:
        experiments = get_experiments_by_phase(args.phase)
        if not experiments:
            raise ValueError(f"Phase {args.phase}에 등록된 실험이 없습니다.")
        print(f"Phase {args.phase} 실험 {len(experiments)}개 실행:")
        for exp in experiments:
            print(f"  - {exp.name}: {exp.description}")
    else:
        raise SystemExit(
            "실행할 실험을 지정하세요: --exp NAME [NAME...] 또는 --phase N\n"
            "등록된 실험 확인: --list"
        )

    # 공통 기준 TrainingConfig
    base_config = TrainingConfig(
        qa_json_path=args.qa_json,
        raw_video_root=args.video_dir,
        label_root=args.label_dir,
        model_id=args.model_id,
        seed=args.seed,
        max_steps=2 if args.smoke_test else -1,
        smoke_test=args.smoke_test,
    )

    results_dir = Path(args.results_dir)

    # 순차 실행
    all_results = []
    for exp in experiments:
        try:
            result = run_experiment(
                exp,
                base_config=base_config,
                results_dir=results_dir,
                checkpoint_root=args.checkpoint_root,
                force_rerun=args.force_rerun,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n[{exp.name}] 오류 발생: {e}")
            import traceback
            traceback.print_exc()

    # 최종 비교 출력
    if len(all_results) > 1:
        print_comparison(results_dir)


if __name__ == "__main__":
    main()
