from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """학습 설정을 담는 데이터 클래스"""
    # 경로
    qa_json_path: str = "data/qa/qa_dataset.json"
    raw_video_root: str = "data/working/pipeline_subset/raw"
    label_root: str = "data/working/pipeline_subset/label"
    checkpoint_dir: str = "checkpoints/Qwen3_VL_4B_Instruct"

    # 모델
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"

    # LoRA
    lora_r: int = 16           # Rank (LoRA 차원)
    lora_alpha: int = 32       # Scaling factor (LoRA 학습률 조절)
    lora_dropout: float = 0.05 # Dropout 확률 (과적합 방지)
    lora_bias: str = "none"    # "none" / "all" / "lora_only"

    # LoRA를 적용할 레이어 이름 패턴. 모델 구조에 따라 조정 필요.
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", # attention 내 투영 레이어
        "gate_proj", "up_proj", "down_proj",    # feed-forward 내 레이어
    ])

    # 학습
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 1 # GPU 메모리에 따라 1 또는 2로 조정
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4 # 배치 사이즈가 1이라서 생기는 불안정한 그래디언트를 보완하기 위해, 4 스텝마다 그래디언트를 누적하여 역전파
    learning_rate: float = 2e-4          # LoRA 학습이므로 풀파인튜닝(ex: 1e-5)보다 상대적으로 높은 학습률을 씁니다.
    warmup_ratio: float = 0.05           # 초기 학습 불안정을 막기 위해 5%의 구간 동안 학습률을 서서히 올립니다.
    weight_decay: float = 0.01           # 가중치가 너무 커지지 않도록 억제하는 정규화
    lr_scheduler_type: str = "cosine"    # 학습률 스케줄러 유형 (cosine, linear 등) - 학습률을 점진적으로 감소시켜 안정적인 수렴 유도

    # 메모리 최적화
    gradient_checkpointing: bool = True # 그래디언트 체크포인팅 활성화 여부. 활성화하면 메모리 사용량이 줄어들지만, 역전파 시 계산량이 늘어나 학습 속도가 느려질 수 있습니다.
    # BF16 (Bfloat16)은 FP16보다 표현 가능한 숫자의 범위가 넓어 Loss가 NaN이 되는(터지는) 현상을 잘 막아줍니다.
    bf16: bool = True
    fp16: bool = False

    # 비디오 입력
    fps: float = 1.0
    max_pixels: int = 360 * 420 # 너무 크면 OOM 발생
    max_seq_len: int = 512      # 텍스트 토큰 최대 길이

    # 분할 (video 단위 7:2:1)
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    # test 비율은 1 - train_ratio - val_ratio (기본 0.1)
    seed: int = 42
    question_types: list[str] | None = None

    # 로깅
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2            # 최근 2개 체크포인트만 저장 (이전 체크포인트는 자동 삭제)
    load_best_model_at_end: bool = True
    report_to: str = "none"              # "none" / "wandb" / "tensorboard"
    run_name: str = "qwen3_vl_4b_traffic_lora"
    dataloader_num_workers: int = 4

    # 시스템 프롬프트 (experiment_config에서 오버라이드)
    system_prompt: str = (
        "당신은 교통사고 영상을 분석하는 전문 분석가입니다. "
        "제공된 블랙박스 영상을 주의 깊게 관찰하고, "
        "사고 상황에 대한 질문에 정확하고 간결하게 답변하세요."
    )

    # 실험 식별자 (experiment_runner가 설정)
    experiment_name: str = ""

    # 파일럿 모드: 카테고리(case_code)별 최대 샘플 수 제한. None이면 전체 사용
    max_samples_per_category: int | None = None

    # CLI 전용 제어 플래그 (run_training에는 영향 없음)
    max_steps: int = -1        # -1이면 num_train_epochs 기준으로 전체 학습. --smoke-test 시 자동으로 2로 설정됨
    no_auto_gpu: bool = False  # True면 _detect_gpu_settings()의 자동 오버라이드를 건너뜀. bf16/fp16/fps/max_pixels를 직접 제어하고 싶을 때 사용
    smoke_test: bool = False   # True면 별도 임시 모델로 2 스텝만 실행해 파이프라인 전체를 빠르게 검증 (OOM·코드 오류 사전 점검용)
