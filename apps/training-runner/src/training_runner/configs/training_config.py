from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """학습 설정을 담는 데이터 클래스."""

    # 로컬 경로
    qa_json_path: str = "data/qa/qa_dataset.json"
    raw_video_root: str = "data/working/pipeline_subset/raw"
    label_root: str = "data/working/pipeline_subset/label"
    checkpoint_dir: str = "checkpoints/Qwen3_VL_4B_Instruct"
    results_dir: str = "outputs/experiments"
    download_root: str = "data/downloads/qwen3_vl"

    # 원격 URL (Colab/NAS HTTP 다운로드용)
    raw_url: str = ""
    label_url: str = ""
    qa_json_url: str = ""

    # 모델
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    prefer_flash_attention: bool = True

    # LoRA
    # [수정] MLP까지 LoRA를 거는 대신 attention projection 중심으로 줄여 속도/안정성을 우선합니다.
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])

    # 학습
    # [수정] 3,600개 영상(≈ 21,600 QA) 기준으로 5 epoch는 다소 무거워 3 epoch + early stopping 쪽이 효율적입니다.
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    # 메모리/속도 최적화
    # [수정] A100 기준 속도 최적화를 위해 기본값을 checkpointing off로 둡니다.
    gradient_checkpointing: bool = False
    bf16: bool = True
    fp16: bool = False
    group_samples_by_video: bool = True
    precache_train_videos: bool = False
    precache_eval_videos: bool = False

    # 비디오 입력
    # [수정] 10초 영상, 독립 질문 태스크 기준으로 0.5 fps / 320x320를 기본값으로 둡니다.
    fps: float = 0.5
    max_pixels: int = 320 * 320
    # [수정] partial truncation 위험을 줄이기 위해 기본 max_seq_len을 512로 상향합니다.
    max_seq_len: int = 512

    # 분할
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    seed: int = 42
    question_types: list[str] | None = None

    # 로깅/저장
    logging_steps: int = 20
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    report_to: str = "none"
    run_name: str = "qwen3_vl_4b_traffic_lora"
    # [수정] 같은 video_id QA를 붙여 처리할 때 Python 메모리 캐시 재사용률을 높이기 위해 기본 worker를 0으로 둡니다.
    dataloader_num_workers: int = 0
    dataloader_prefetch_factor: int = 2
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    early_stopping_patience: int = 1
    early_stopping_threshold: float = 0.0
    generation_max_new_tokens: int = 64

    # [수정] 최종 평가는 기본적으로 BERTScore를 끄고 필요 시 CLI로만 활성화해 평가 병목을 줄입니다.
    compute_bertscore: bool = False
    bertscore_batch_size: int = 32

    # 시스템 프롬프트
    system_prompt: str = (
        "당신은 교통사고 영상을 분석하는 전문 분석가입니다. "
        "제공된 블랙박스 영상을 주의 깊게 관찰하고, "
        "사고 상황에 대한 질문에 정확하고 간결하게 답변하세요."
    )

    experiment_name: str = ""
    max_samples_per_category: int | None = None
    video_cache_dir: str = ""

    # CLI 제어
    max_steps: int = -1
    no_auto_gpu: bool = False
    smoke_test: bool = False
    force_download: bool = False
