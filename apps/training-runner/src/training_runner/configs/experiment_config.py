"""
실험 카탈로그 — Phase 0 / 1 / 2

실험 설계 문서(실험설계.pdf)에 정의된 모든 실험 조건을 코드로 표현합니다.
각 ExperimentConfig는 TrainingConfig의 필드를 오버라이드하는 방식으로 동작합니다.

사용 예:
    from training_runner.configs import ExperimentConfig, EXPERIMENT_CATALOG
    exp = EXPERIMENT_CATALOG["A-1"]
    config = exp.build_training_config(base_qa_json="data/qa/qa_dataset.json", ...)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from training_runner.configs.training_config import TrainingConfig

# ---------------------------------------------------------------------------
# 시스템 프롬프트 상수 (실험 C 계열)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {
    # C-1: ablation — 시스템 프롬프트 없음 (None으로 처리)
    "none": "",

    # C-2: 현재 기본값 (일반 분석 전문가)
    "default": (
        "당신은 교통사고 영상을 분석하는 전문 분석가입니다. "
        "제공된 블랙박스 영상을 주의 깊게 관찰하고, "
        "사고 상황에 대한 질문에 정확하고 간결하게 답변하세요."
    ),

    # C-3: case_code 컨텍스트 힌트 포함
    # {case_code} 플레이스홀더는 데이터셋 로딩 시 카테고리 이름으로 대체됩니다.
    "case_code_context": (
        "당신은 교통사고 영상을 분석하는 전문 분석가입니다. "
        "이 영상의 사고 유형 코드는 '{case_code}'입니다. "
        "제공된 블랙박스 영상을 주의 깊게 관찰하고, "
        "사고 상황에 대한 질문에 정확하고 간결하게 답변하세요."
    ),

    # C-4: 한국 도로교통법 조항 요약 포함 (법률 지식 주입)
    "traffic_law": (
        "당신은 한국 도로교통법 전문가이자 교통사고 분석가입니다.\n"
        "【관련 법령 요약】\n"
        "- 도로교통법 제13조: 차마는 도로의 중앙 우측을 통행해야 합니다.\n"
        "- 도로교통법 제25조: 교차로 통행 방법 — 직진 차량이 우선입니다.\n"
        "- 도로교통법 제26조: 신호에 따른 통행 의무 — 신호위반 시 과실 가중.\n"
        "- 도로교통법 제38조: 차선 변경 시 방향지시등 의무.\n"
        "- 과실비율 산정 시 우선순위: ① 신호위반 ② 중앙선 침범 ③ 속도위반 ④ 안전거리 미확보.\n"
        "제공된 블랙박스 영상을 분석하고, 위 법령을 근거로 정확하게 답변하세요."
    ),
}


# ---------------------------------------------------------------------------
# ExperimentConfig 데이터클래스
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """단일 실험 설정.

    TrainingConfig의 기본값에서 변경할 필드만 overrides에 지정합니다.
    build_training_config()로 실제 TrainingConfig를 생성합니다.
    """
    name: str                      # "A-1", "B-2", "C-3", "phase0_no_prompt" 등
    phase: int                     # 0, 1, 2
    description: str               # 실험 목적 한 줄 설명

    # TrainingConfig 필드 오버라이드 (지정하지 않으면 TrainingConfig 기본값 사용)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    fps: float = 1.0
    max_pixels: int = 360 * 420
    learning_rate: float = 2e-4
    num_train_epochs: int = 5
    warmup_ratio: float = 0.05
    eval_steps: int = 50

    # 프롬프트 설정
    system_prompt_key: str = "default"   # SYSTEM_PROMPTS 딕셔너리 키
    use_no_system_prompt: bool = False   # True면 system_prompt를 빈 문자열로 설정

    # Phase 제어
    skip_training: bool = False          # True면 학습 없이 평가만 (Phase 0)

    # 파일럿 데이터 제한 (Phase 1: case_code별 N개)
    max_samples_per_category: int | None = None

    # 분할 비율 (Phase 1 파일럿은 6:2:2)
    train_ratio: float = 0.7
    val_ratio: float = 0.2

    # 추가 메타
    tags: list[str] = field(default_factory=list)

    @property
    def system_prompt(self) -> str:
        if self.use_no_system_prompt:
            return ""
        return SYSTEM_PROMPTS.get(self.system_prompt_key, SYSTEM_PROMPTS["default"])

    @property
    def checkpoint_subdir(self) -> str:
        """체크포인트 저장 하위 디렉토리 이름."""
        return f"phase{self.phase}_{self.name}"

    def build_training_config(
        self,
        base_config: TrainingConfig | None = None,
        checkpoint_root: str = "checkpoints",
    ) -> TrainingConfig:
        """이 실험 설정을 적용한 TrainingConfig를 생성합니다.

        Args:
            base_config:      기준 TrainingConfig. None이면 기본값 사용.
            checkpoint_root:  체크포인트 루트 디렉토리.

        Returns:
            실험 오버라이드가 적용된 TrainingConfig
        """
        cfg = base_config or TrainingConfig()

        cfg.lora_r = self.lora_r
        cfg.lora_alpha = self.lora_alpha
        cfg.lora_dropout = self.lora_dropout
        cfg.fps = self.fps
        cfg.max_pixels = self.max_pixels
        cfg.learning_rate = self.learning_rate
        cfg.num_train_epochs = self.num_train_epochs
        cfg.warmup_ratio = self.warmup_ratio
        cfg.eval_steps = self.eval_steps
        cfg.train_ratio = self.train_ratio
        cfg.val_ratio = self.val_ratio
        cfg.max_samples_per_category = self.max_samples_per_category
        cfg.system_prompt = self.system_prompt
        cfg.experiment_name = self.name
        cfg.checkpoint_dir = f"{checkpoint_root}/{self.checkpoint_subdir}"
        cfg.run_name = f"qwen3vl_{self.name}"

        return cfg


# ---------------------------------------------------------------------------
# 실험 카탈로그 정의
# ---------------------------------------------------------------------------

def _make_catalog() -> dict[str, ExperimentConfig]:
    catalog: dict[str, ExperimentConfig] = {}

    # -----------------------------------------------------------------------
    # Phase 0 — 제로샷 베이스라인 (학습 없음)
    # -----------------------------------------------------------------------
    catalog["phase0_no_prompt"] = ExperimentConfig(
        name="phase0_no_prompt",
        phase=0,
        description="제로샷 베이스라인 — 시스템 프롬프트 없음",
        skip_training=True,
        use_no_system_prompt=True,
        tags=["phase0", "baseline"],
    )
    catalog["phase0_default_prompt"] = ExperimentConfig(
        name="phase0_default_prompt",
        phase=0,
        description="제로샷 베이스라인 — 기본 시스템 프롬프트 있음",
        skip_training=True,
        system_prompt_key="default",
        tags=["phase0", "baseline"],
    )

    # -----------------------------------------------------------------------
    # Phase 1A — LoRA rank 탐색 (case_code별 10개, 6:2:2 분할)
    # -----------------------------------------------------------------------
    _pilot_kw = dict(
        phase=1,
        max_samples_per_category=10,
        train_ratio=0.6,
        val_ratio=0.2,
        num_train_epochs=5,
        tags=["phase1", "lora_rank"],
    )
    catalog["A-1"] = ExperimentConfig(
        name="A-1", description="LoRA rank=8  (작은 용량)",
        lora_r=8,  lora_alpha=16,  lora_dropout=0.05,
        **_pilot_kw,
    )
    catalog["A-2"] = ExperimentConfig(
        name="A-2", description="LoRA rank=16 (기본값)",
        lora_r=16, lora_alpha=32,  lora_dropout=0.05,
        **_pilot_kw,
    )
    catalog["A-3"] = ExperimentConfig(
        name="A-3", description="LoRA rank=32 (큰 용량)",
        lora_r=32, lora_alpha=64,  lora_dropout=0.05,
        **_pilot_kw,
    )
    catalog["A-4"] = ExperimentConfig(
        name="A-4", description="LoRA rank=64 (최대 용량)",
        lora_r=64, lora_alpha=128, lora_dropout=0.10,
        **_pilot_kw,
    )

    # -----------------------------------------------------------------------
    # Phase 1B — 입력 해상도/fps 탐색
    # -----------------------------------------------------------------------
    _pilot_kw_b = dict(
        phase=1,
        max_samples_per_category=10,
        train_ratio=0.6,
        val_ratio=0.2,
        num_train_epochs=5,
        tags=["phase1", "fps_resolution"],
    )
    catalog["B-1"] = ExperimentConfig(
        name="B-1", description="fps=0.5 max_pixels=320×320 (최소 입력)",
        fps=0.5, max_pixels=320 * 320,
        **_pilot_kw_b,
    )
    catalog["B-2"] = ExperimentConfig(
        name="B-2", description="fps=1.0 max_pixels=360×420 (기본값)",
        fps=1.0, max_pixels=360 * 420,
        **_pilot_kw_b,
    )
    catalog["B-3"] = ExperimentConfig(
        name="B-3", description="fps=2.0 max_pixels=360×420 (시간적 정보 증가)",
        fps=2.0, max_pixels=360 * 420,
        **_pilot_kw_b,
    )

    # -----------------------------------------------------------------------
    # Phase 1C — 프롬프트 전략
    # -----------------------------------------------------------------------
    _pilot_kw_c = dict(
        phase=1,
        max_samples_per_category=10,
        train_ratio=0.6,
        val_ratio=0.2,
        num_train_epochs=5,
        tags=["phase1", "prompt_strategy"],
    )
    catalog["C-1"] = ExperimentConfig(
        name="C-1", description="프롬프트 없음 (ablation)",
        use_no_system_prompt=True,
        **_pilot_kw_c,
    )
    catalog["C-2"] = ExperimentConfig(
        name="C-2", description="기본 시스템 프롬프트 (일반 분석 전문가)",
        system_prompt_key="default",
        **_pilot_kw_c,
    )
    catalog["C-3"] = ExperimentConfig(
        name="C-3", description="case_code 컨텍스트 포함 (도메인 힌트)",
        system_prompt_key="case_code_context",
        **_pilot_kw_c,
    )
    catalog["C-4"] = ExperimentConfig(
        name="C-4", description="한국 도로교통법 조항 요약 포함 (법률 지식 주입)",
        system_prompt_key="traffic_law",
        **_pilot_kw_c,
    )

    # -----------------------------------------------------------------------
    # Phase 2 — 본 학습 (Phase 1 최적 설정 적용, 전체 데이터)
    # 실제 사용 시 Phase 1 결과를 보고 lora_r / learning_rate를 조정하세요.
    # -----------------------------------------------------------------------
    catalog["phase2_main"] = ExperimentConfig(
        name="phase2_main",
        phase=2,
        description="본 학습 — Phase 1 최적 설정 적용 (80 클래스, 전체 데이터)",
        lora_r=16,           # Phase 1 결과로 업데이트 예정
        lora_alpha=32,
        lora_dropout=0.05,
        fps=1.0,
        max_pixels=360 * 420,
        learning_rate=2e-4,
        num_train_epochs=5,
        warmup_ratio=0.03,
        eval_steps=100,
        train_ratio=0.7,
        val_ratio=0.2,
        max_samples_per_category=200,   # soft capping: 다수 클래스 최대 200개
        system_prompt_key="default",
        tags=["phase2", "main_training"],
    )

    return catalog


EXPERIMENT_CATALOG: dict[str, ExperimentConfig] = _make_catalog()


# ---------------------------------------------------------------------------
# 조회 유틸
# ---------------------------------------------------------------------------

def get_experiments_by_phase(phase: int) -> list[ExperimentConfig]:
    """지정된 phase의 모든 실험을 반환합니다."""
    return [exp for exp in EXPERIMENT_CATALOG.values() if exp.phase == phase]


def get_experiments_by_tag(tag: str) -> list[ExperimentConfig]:
    """특정 태그를 가진 실험 목록을 반환합니다."""
    return [exp for exp in EXPERIMENT_CATALOG.values() if tag in exp.tags]


def list_experiments() -> None:
    """등록된 모든 실험을 표 형식으로 출력합니다."""
    print(f"{'이름':<25} {'Phase':<7} {'설명'}")
    print("-" * 80)
    for name, exp in sorted(EXPERIMENT_CATALOG.items(), key=lambda x: (x[1].phase, x[0])):
        skip_tag = " [skip_train]" if exp.skip_training else ""
        print(f"{name:<25} Phase {exp.phase}  {exp.description}{skip_tag}")
