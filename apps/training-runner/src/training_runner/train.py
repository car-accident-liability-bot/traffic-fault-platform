from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from traffic_ai_core.Qwen3_VL_4B_Instruct.model.model import load_model
from training_runner.configs import TrainingConfig
from training_runner.dataset import TrafficAccidentQADataset, build_dataloaders
from training_runner.evaluation.evaluator import diagnose_gradient_flow


class _GradDiagCallback(TrainerCallback):
    """첫 번째 backward 이후 gradient flow를 1회 진단합니다."""

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        if state.global_step == 1:
            print("\n[Gradient 흐름 진단 — step 1]")
            diagnose_gradient_flow(model)


def run_training(config: TrainingConfig | None = None) -> None:
    if config is None:
        config = TrainingConfig()

    _print_config(config)

    print("\n[0/7] 데이터 파일 검증 중...")
    _validate_data_files(config)

    print("\n[1/7] 모델 로드 중...")
    # 원본 Qwen 모델과 프로세서를 불러옵니다. (보통 8비트/4비트 양자화 상태로 불러오기도 합니다)
    base_model, processor = load_model(config.model_id)

    # [매우 중요한 핵심 버그 방지 로직]
    # enable_input_require_grads는 반드시 get_peft_model 이전에 호출해야 함
    # 순서가 바뀌면 frozen 레이어를 통한 gradient 전파가 끊겨 LoRA 가중치가 학습되지 않음
    if config.gradient_checkpointing:
        print("[2/7] Gradient Checkpointing 활성화...")
        base_model.enable_input_require_grads()

    print("[3/7] LoRA 적용 중...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    print("\n[4/7] 데이터셋 구성 중...")
    # 테스트셋 인자로 넘겨서 평가지표 계산해야함
    train_ds, val_ds, test_ds, *_ = build_dataloaders(
        qa_json_path=config.qa_json_path,
        raw_video_root=config.raw_video_root,
        label_root=config.label_root,
        processor=processor,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
        batch_size=config.per_device_train_batch_size,
        fps=config.fps,
        max_pixels=config.max_pixels,
        max_seq_len=config.max_seq_len,
        question_types=config.question_types,
        num_workers=config.dataloader_num_workers,
        system_prompt=config.system_prompt,
        max_samples_per_category=config.max_samples_per_category,
        video_cache_dir=config.video_cache_dir or None,
    )

    dist = train_ds.get_distribution()
    print("\n[학습 데이터 분포]")
    for cat, cnt in sorted(dist["by_category"].items()):
        print(f"  {cat:<30} {cnt:>4}샘플  {'█' * (cnt // 5)}")
    print(f"  {'합계':<30} {dist['total']:>4}샘플 ({dist['unique_videos']}개 비디오)")

    print("\n[4.5/7] 라벨 마스킹 검증 중...")
    _verify_label_masking(train_ds, processor)

    print("\n[5/7] TrainingArguments 구성 중...")
    checkpoint_path = Path(config.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_path),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_dir=str(checkpoint_path / "logs"),
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=config.report_to,
        run_name=config.run_name,
        dataloader_num_workers=config.dataloader_num_workers,
        max_steps=config.max_steps,    # -1이면 num_train_epochs 전체; smoke-test 시 2로 설정해 빠른 파이프라인 검증
        remove_unused_columns=False,  # 시각 텐서(pixel_values 등)를 Trainer가 제거하지 않도록
        label_names=["labels"],       # Trainer가 loss 계산에 labels 키를 인식하도록
    )

    if config.smoke_test:
        print("\n[5.5/7] Smoke test 실행 중...")
        _run_smoke_test(config, train_ds)

    print("\n[6/7] 학습 시작...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=TrafficAccidentQADataset.collate_fn,
        callbacks=[_GradDiagCallback()],
    )
    trainer.train()

    print("\n[7/7] 어댑터 저장 중...")
    final_dir = checkpoint_path / "final_adapter"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"  저장 완료: {final_dir}")
    _finalize_checkpoint(final_dir)

    metrics = trainer.evaluate(eval_dataset=test_ds)
    print(f"\n  eval_loss: {metrics.get('eval_loss', 'N/A'):.4f}")
    print("\n========== 학습 완료 ==========")

    # === 상세 평가 (방어 코드 포함) ===
    print("\n + 상세 평가 수행 중...")

    # 1. 패키지 체크
    try:
        from traffic_metrics import QAEvaluator
        from training_runner.evaluation import convert_trainer_predictions
    except ImportError as e:
        print(f"⚠️  평가 모듈 import 실패: {e}")
        print("   → pip install -e packages/traffic-metrics")
        print("\n========== 평가 건너뜀 ==========")
        return

    # 2. 테스트셋 체크
    if test_ds is None or len(test_ds) == 0:
        print("⚠️  테스트셋이 비어있어 상세 평가를 건너뜁니다.")
        print("\n========== 평가 건너뜀 ==========")
        return

    # 3. 추론 (OOM 방어)
    try:
        test_predictions = trainer.predict(test_ds)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠️  GPU 메모리 부족으로 상세 평가 실패")
            print("   → per_device_eval_batch_size를 줄여보세요")
            import torch
            torch.cuda.empty_cache()
            print("\n========== 평가 실패 (학습은 성공) ==========")
            return
        else:
            raise

    # 4. 변환 및 평가
    try:
        evaluation_data = convert_trainer_predictions(
            test_predictions, test_ds, processor
        )

        evaluator = QAEvaluator()
        detailed_results = evaluator.evaluate(evaluation_data)
    except Exception as e:
        print(f"⚠️  평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        print("\n========== 평가 실패 (학습은 성공) ==========")
        return

    # 5. 콘솔 출력
    print(f"\n=== 상세 평가 결과 ===")
    print(f"Macro F1: {detailed_results['overall']['macro_f1']:.4f}")
    print(f"총 샘플: {detailed_results['overall']['total_samples']}")
    print(f"정답률: {detailed_results['overall']['accuracy']:.2%}")

    print(f"\n[타입별 F1 Score]")
    for q_type, type_metrics in detailed_results['per_type'].items():
        f1 = type_metrics['f1']
        support = type_metrics['support']
        print(f"  {q_type:<25} F1={f1:.3f} (n={support})")

        # 특수 메트릭 표시
        if 'recall_highlighted' in type_metrics:
            print(f"    → Recall={type_metrics['recall']:.3f}")
        if 'mae' in type_metrics:
            print(f"    → MAE={type_metrics['mae']:.2f}")
        if 'accuracy' in type_metrics:
            print(f"    → Accuracy={type_metrics['accuracy']:.3f}")

    # 6. 파일 저장 (checkpoint 디렉토리에만)
    try:
        import json
        from pathlib import Path

        output_path = Path(config.checkpoint_dir) / "evaluation_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 평가 결과 저장: {output_path}")
    except Exception as e:
        print(f"⚠️  결과 저장 실패: {e}")

    # 7. WandB 로깅 (안전)
    if config.report_to == "wandb":
        try:
            import wandb
            wandb.log({
                "test/macro_f1": detailed_results['overall']['macro_f1'],
                "test/accuracy": detailed_results['overall']['accuracy'],
                "test/total_samples": detailed_results['overall']['total_samples'],
            })
            for q_type, type_metrics in detailed_results['per_type'].items():
                wandb.log({f"test/{q_type}/f1": type_metrics['f1']})
            print("✅ WandB 로깅 완료!")
        except Exception as e:
            print(f"⚠️  WandB 로깅 실패: {e}")

    print("\n========== 평가 완료 ==========")


def _validate_data_files(config: TrainingConfig) -> None:
    """학습 전 QA JSON / 원본 MP4 / 라벨 JSON 파일의 3방향 매칭을 확인한다.

    매칭 결과가 0개면 경로 설정이 잘못된 것이므로 RuntimeError를 발생시킨다.
    """
    import json

    with open(config.qa_json_path, encoding="utf-8") as f:
        qa_data = json.load(f)

    raw_mp4s    = list(Path(config.raw_video_root).rglob("*.mp4"))
    label_jsons = list(Path(config.label_root).rglob("*.json"))
    raw_stems   = {p.stem for p in raw_mp4s}
    label_stems = {p.stem for p in label_jsons}
    qa_stems    = {Path(e["video_id"]).stem for e in qa_data}
    matched     = qa_stems & raw_stems & label_stems

    print(f"  Raw 비디오 : {len(raw_mp4s)}개")
    print(f"  Label JSON : {len(label_jsons)}개")
    print(f"  QA 항목    : {len(qa_data)}개")
    print(f"  3방향 매칭 : {len(matched)}개")

    if len(matched) == 0:
        raise RuntimeError("3방향 매칭 결과가 0개입니다. 경로 설정을 확인하세요.")


def _verify_label_masking(train_ds, processor) -> None:
    """첫 번째 학습 샘플의 SFT 라벨 마스킹(-100 구간)이 올바른지 확인한다.

    답변 구간이 0이면 데이터셋 전처리 버그일 가능성이 높으므로 경고를 출력한다.
    """
    sample      = train_ds[0]
    input_ids   = sample["input_ids"]
    labels      = sample["labels"]
    answer_mask = labels != -100

    print(f"  전체 시퀀스 : {len(input_ids)} 토큰")
    print(f"  마스킹 구간 : {(~answer_mask).sum().item()} 토큰 (-100)")
    print(f"  답변 구간   : {answer_mask.sum().item()} 토큰")

    if answer_mask.sum().item() == 0:
        print("  ⚠ 경고: 답변 구간이 0입니다. 데이터셋 전처리를 확인하세요.")
        return

    decoded = processor.tokenizer.decode(input_ids[answer_mask], skip_special_tokens=True)
    print(f"  디코딩 답변 : {decoded!r}")


def _run_smoke_test(config: TrainingConfig, train_ds) -> None:
    """별도 임시 모델+Trainer로 2 스텝만 실행해 파이프라인 전체를 빠르게 검증한다.

    완료 후 메모리를 즉시 해제해 본 학습에서 OOM이 발생하지 않도록 한다.
    """
    print("  (별도 임시 모델로 2 스텝 실행 중...)")
    smoke_base, _ = load_model(config.model_id)
    if config.gradient_checkpointing:
        smoke_base.enable_input_require_grads()

    smoke_model = get_peft_model(smoke_base, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
    ))

    Trainer(
        model=smoke_model,
        args=TrainingArguments(
            output_dir="/tmp/smoke_test",
            max_steps=2,
            per_device_train_batch_size=1,
            bf16=config.bf16,
            fp16=config.fp16,
            gradient_checkpointing=config.gradient_checkpointing,
            remove_unused_columns=False,
            label_names=["labels"],
            report_to="none",
            logging_steps=1,
        ),
        train_dataset=train_ds,
        data_collator=TrafficAccidentQADataset.collate_fn,
    ).train()

    print("  ✓ Smoke test 통과")
    del smoke_model, smoke_base
    torch.cuda.empty_cache()


def _finalize_checkpoint(final_dir: Path) -> None:
    """저장된 어댑터 파일 목록을 출력하고, safetensors를 best.pt로 변환 저장한다."""
    from safetensors.torch import load_file

    print("\n[체크포인트 파일 목록]")
    for f in sorted(final_dir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")

    safetensors_path = final_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        weights = load_file(str(safetensors_path))
        best_pt = final_dir.parent / "best.pt"
        torch.save(weights, str(best_pt))
        print(f"\n  best.pt 저장: {best_pt} ({best_pt.stat().st_size / 1e6:.1f} MB, {len(weights)}개 파라미터)")


def _print_config(config: TrainingConfig) -> None:
    print("=" * 60)
    print("Qwen3-VL-4B LoRA 미세조정 설정")
    print("-" * 60)
    print(f"  모델              : {config.model_id}")
    print(f"  LoRA rank / alpha : {config.lora_r} / {config.lora_alpha}")
    print(f"  학습률            : {config.learning_rate}")
    print(f"  에폭              : {config.num_train_epochs}")
    print(f"  배치 크기         : {config.per_device_train_batch_size} × {config.gradient_accumulation_steps}")
    print(f"  fps / max_pixels  : {config.fps} / {config.max_pixels}")
    print(f"  bf16 / fp16       : {config.bf16} / {config.fp16}")
    print(f"  체크포인트        : {config.checkpoint_dir}")
    if config.experiment_name:
        print(f"  실험 이름         : {config.experiment_name}")
    if config.max_samples_per_category is not None:
        print(f"  카테고리별 최대   : {config.max_samples_per_category}개 (파일럿 모드)")
    gpu_info = (
        f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB)"
        if torch.cuda.is_available() else "CPU"
    )
    print(f"  GPU               : {gpu_info}")
    print("=" * 60)


def _parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Qwen3-VL-4B 교통사고 QA LoRA 미세조정")
    d = TrainingConfig()

    parser.add_argument("--qa-json",    type=str,   default=d.qa_json_path)
    parser.add_argument("--video-dir",  type=str,   default=d.raw_video_root)
    parser.add_argument("--label-dir",  type=str,   default=d.label_root)
    parser.add_argument("--output-dir", type=str,   default=d.checkpoint_dir)
    parser.add_argument("--model-id",   type=str,   default=d.model_id)
    parser.add_argument("--lora-r",     type=int,   default=d.lora_r)
    parser.add_argument("--lora-alpha", type=int,   default=d.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=d.lora_dropout)
    parser.add_argument("--epochs",     type=int,   default=d.num_train_epochs)
    parser.add_argument("--lr",         type=float, default=d.learning_rate)
    parser.add_argument("--batch-size", type=int,   default=d.per_device_train_batch_size)
    parser.add_argument("--grad-accum", type=int,   default=d.gradient_accumulation_steps)
    parser.add_argument("--fps",        type=float, default=d.fps)
    parser.add_argument("--max-pixels", type=int,   default=d.max_pixels)
    parser.add_argument("--max-seq-len",type=int,   default=d.max_seq_len)
    parser.add_argument("--fp16",       action="store_true")
    parser.add_argument("--no-bf16",    action="store_true")
    parser.add_argument("--seed",       type=int,   default=d.seed)
    parser.add_argument("--report-to",  type=str,   default=d.report_to,
                        choices=["none", "wandb", "tensorboard"])
    parser.add_argument("--max-steps",  type=int,   default=d.max_steps,
                        help="학습 최대 스텝 수. -1이면 num_train_epochs 전체 학습")
    parser.add_argument("--smoke-test", action="store_true",
                        help="max_steps=2로 강제 설정해 forward/backward 동작만 빠르게 확인 (OOM·코드 오류 사전 점검용)")
    parser.add_argument("--no-auto-gpu", action="store_true",
                        help="GPU VRAM 기반 자동 설정(bf16/fp16/fps/max_pixels) 비활성화. 수동으로 정밀도·해상도를 지정할 때 사용")

    args = parser.parse_args()
    return TrainingConfig(
        qa_json_path=args.qa_json,
        raw_video_root=args.video_dir,
        label_root=args.label_dir,
        checkpoint_dir=args.output_dir,
        model_id=args.model_id,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        fps=args.fps,
        max_pixels=args.max_pixels,
        max_seq_len=args.max_seq_len,
        bf16=not args.no_bf16,
        fp16=args.fp16,
        seed=args.seed,
        report_to=args.report_to,
        max_steps=2 if args.smoke_test else args.max_steps,
        no_auto_gpu=args.no_auto_gpu,
        smoke_test=args.smoke_test,
    )


def _detect_gpu_settings() -> dict:
    """GPU VRAM을 감지해 권장 학습 설정을 반환한다.

    A100/H100(VRAM ≥ 35GB)은 bf16 + 고해상도로, T4/V100은 fp16 + 저해상도로 설정한다.
    --no-auto-gpu 플래그를 주면 main()에서 이 함수를 건너뛴다.
    """
    if not torch.cuda.is_available():
        print("GPU 없음 (CPU 실행 — 디버깅 전용)")
        return {}

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU  : {gpu_name}")
    print(f"VRAM : {vram_gb:.1f} GB")

    if vram_gb >= 35:
        # A100/H100: bf16은 fp16보다 수치 안정성이 높고 Loss NaN 발생이 적다
        print("→ A100/H100 감지: bf16=True, fps=1.0, max_pixels=360×420 권장")
        return {"bf16": True, "fp16": False, "fps": 1.0, "max_pixels": 360 * 420}
    else:
        # T4/V100: bf16을 지원하지 않거나 불안정하므로 fp16으로 전환하고
        # fps·max_pixels을 낮춰 16GB VRAM 안에서 OOM 없이 학습할 수 있게 함
        print("→ T4/V100 감지: fp16=True, fps=0.5, max_pixels=320×320 자동 적용")
        return {"bf16": False, "fp16": True, "fps": 0.5, "max_pixels": 320 * 320}


def main() -> None:
    config = _parse_args()

    # GPU 자동 감지: T4/V100에서는 fp16·저해상도로 덮어쓴다.
    # 정밀도나 해상도를 직접 제어하려면 --no-auto-gpu 플래그를 추가하면 됨
    if not config.no_auto_gpu:
        gpu_settings = _detect_gpu_settings()
        for key, val in gpu_settings.items():
            setattr(config, key, val)

    run_training(config)


if __name__ == "__main__":
    main()
