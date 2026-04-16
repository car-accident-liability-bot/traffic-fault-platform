from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests
import torch
from bs4 import BeautifulSoup
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Sampler
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from traffic_ai_core.Qwen3_VL_4B_Instruct.model.model import load_model
from training_runner.configs import TrainingConfig
from training_runner.dataset import TrafficAccidentQADataset, build_dataloaders
from training_runner.evaluation.evaluator import EvaluatorConfig, diagnose_gradient_flow, run_evaluation

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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


class VideoGroupedSampler(Sampler[int]):
    """같은 video_id의 QA 샘플을 붙여서 순회해 프레임 캐시 효율을 높입니다.

    [수정]
    - per_qa를 유지하면서도 "같은 영상 QA 6개가 붙어 나오게" 만들어
      프레임 디코딩/로드가 1회 캐시 후 재사용되도록 유도합니다.
    """

    def __init__(self, dataset: TrafficAccidentQADataset, *, seed: int = 42, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.seed = seed
        self.shuffle = shuffle
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        groups = [list(group) for group in self.dataset.grouped_indices_by_video()]
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        if self.shuffle:
            rng.shuffle(groups)

        for group in groups:
            # [수정] 그룹 내부도 섞어 질문 순서 고정 overfitting을 줄입니다.
            if self.shuffle and len(group) > 1:
                rng.shuffle(group)
            for index in group:
                yield index


class VideoAwareTrainer(Trainer):
    """[수정] train dataloader에 video-grouped sampler를 꽂기 위한 Trainer 서브클래스."""

    def __init__(self, *args, group_samples_by_video: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_samples_by_video = group_samples_by_video

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        num_workers = int(self.args.dataloader_num_workers)
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": bool(num_workers),
        }
        prefetch_factor = getattr(self.args, "dataloader_prefetch_factor", None)
        if num_workers > 0 and prefetch_factor is not None:
            dataloader_params["prefetch_factor"] = prefetch_factor

        if self.group_samples_by_video and isinstance(self.train_dataset, TrafficAccidentQADataset):
            dataloader_params["sampler"] = VideoGroupedSampler(
                self.train_dataset,
                seed=self.args.seed,
                shuffle=True,
            )
        else:
            dataloader_params["sampler"] = self._get_train_sampler()

        return DataLoader(self.train_dataset, **dataloader_params)


def _http_get_text(url: str, *, timeout: int = 30) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def _list_http_entries(url: str) -> list[tuple[str, bool]]:
    """단순 디렉터리 인덱스 HTML에서 파일/폴더 목록을 읽습니다."""
    html = _http_get_text(url.rstrip("/") + "/")
    soup = BeautifulSoup(html, "html.parser")
    entries: list[tuple[str, bool]] = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href in {"../", "./"}:
            continue
        if href.startswith(("?", "/", "http", "#")):
            continue
        is_dir = href.endswith("/")
        name = href.rstrip("/")
        if name:
            entries.append((name, is_dir))
    return entries


def _collect_http_files(root_url: str, *, suffix: str) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []

    for name, is_dir in _list_http_entries(root_url):
        item_url = f"{root_url.rstrip('/')}/{quote(name)}"
        if is_dir:
            for child_name, child_is_dir in _list_http_entries(item_url):
                if child_is_dir:
                    continue
                if child_name.endswith(suffix):
                    child_url = f"{item_url}/{quote(child_name)}"
                    files.append((child_url, Path(name) / child_name))
        elif name.endswith(suffix):
            files.append((item_url, Path(name)))

    return files


def _download_file(url: str, destination: Path, *, chunk_size: int = 1024 * 1024) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as file_obj:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file_obj.write(chunk)
    os.replace(tmp_path, destination)


def _download_http_tree(
    root_url: str,
    destination_root: Path,
    *,
    suffix: str,
    max_workers: int = 8,
    force: bool = False,
) -> int:
    """[수정] Colab에서 raw/label 파일을 병렬로 로컬 SSD에 내려받습니다."""
    files = _collect_http_files(root_url, suffix=suffix)
    if not files:
        raise RuntimeError(f"다운로드 대상 파일을 찾지 못했습니다: {root_url}")

    tasks: list[tuple[str, Path]] = []
    for url, rel_path in files:
        target = destination_root / rel_path
        if not force and target.exists() and target.stat().st_size > 0:
            continue
        tasks.append((url, target))

    print(f"[download] {root_url} -> {destination_root} | 전체 {len(files)}개 | 신규/갱신 {len(tasks)}개")
    if not tasks:
        return len(files)

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_download_file, url, target) for url, target in tasks]
        for future in futures:
            future.result()
            completed += 1
            if completed == len(tasks) or completed % 50 == 0:
                print(f"  - {completed}/{len(tasks)}")
    return len(files)


def _download_json(url: str, destination: Path, *, force: bool = False) -> None:
    if not force and destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = requests.get(url, timeout=120)
    payload.raise_for_status()
    destination.write_bytes(payload.content)
    print(f"[download] QA JSON 저장: {destination}")


def _prepare_local_data(config: TrainingConfig) -> TrainingConfig:
    """raw-url / label-url / qa-json-url가 주어지면 로컬 경로로 자동 변환합니다."""
    if not (config.raw_url and config.label_url and config.qa_json_url):
        return config

    download_root = Path(config.download_root)
    raw_dir = download_root / "raw"
    label_dir = download_root / "label"
    qa_json_path = download_root / "qa_dataset.json"

    _download_http_tree(config.raw_url, raw_dir, suffix=".mp4", force=config.force_download)
    _download_http_tree(config.label_url, label_dir, suffix=".json", force=config.force_download)
    _download_json(config.qa_json_url, qa_json_path, force=config.force_download)

    config.raw_video_root = str(raw_dir)
    config.label_root = str(label_dir)
    config.qa_json_path = str(qa_json_path)

    # [수정] video cache 경로를 download_root 기준으로 자동 설정해 evaluator와 공유합니다.
    if not config.video_cache_dir:
        config.video_cache_dir = str(download_root / "frame_cache")
    return config


def _group_paths_by_stem(paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[path.stem].append(path)
    return grouped


def _validate_data_files(config: TrainingConfig) -> None:
    with open(config.qa_json_path, encoding="utf-8") as file_obj:
        qa_data = json.load(file_obj)

    if not isinstance(qa_data, list) or not qa_data:
        raise RuntimeError("qa_dataset.json 형식이 올바르지 않거나 비어 있습니다.")

    raw_mp4s = list(Path(config.raw_video_root).rglob("*.mp4"))
    label_jsons = list(Path(config.label_root).rglob("*.json"))
    raw_by_stem = _group_paths_by_stem(raw_mp4s)
    label_by_stem = _group_paths_by_stem(label_jsons)
    qa_video_ids = [str(entry.get("video_id", "")).strip() for entry in qa_data]
    qa_stems = {Path(video_id).stem for video_id in qa_video_ids if video_id}
    raw_stems = set(raw_by_stem.keys())
    label_stems = set(label_by_stem.keys())
    matched = qa_stems & raw_stems & label_stems

    duplicate_raw = {stem: paths for stem, paths in raw_by_stem.items() if len(paths) > 1}
    duplicate_label = {stem: paths for stem, paths in label_by_stem.items() if len(paths) > 1}
    if duplicate_raw or duplicate_label:
        duplicate_msgs: list[str] = []
        for stem, paths in list(sorted(duplicate_raw.items()))[:5]:
            duplicate_msgs.append(f"raw 중복 stem={stem}: {[str(path) for path in paths[:3]]}")
        for stem, paths in list(sorted(duplicate_label.items()))[:5]:
            duplicate_msgs.append(f"label 중복 stem={stem}: {[str(path) for path in paths[:3]]}")
        raise RuntimeError("stem 기준 중복 파일이 있어 데이터 매핑이 모호합니다.\n" + "\n".join(duplicate_msgs))

    seen_video_ids: set[str] = set()
    duplicate_video_ids: list[str] = []
    invalid_entries: list[str] = []
    empty_fields: list[str] = []
    unexpected_question_types: list[str] = []
    qa_count_distribution: Counter[int] = Counter()
    missing_required_qtypes: list[str] = []

    for entry in qa_data:
        video_id = str(entry.get("video_id", "")).strip()
        qa_pairs = entry.get("qa_pairs")
        if not video_id or not isinstance(qa_pairs, list):
            invalid_entries.append(str(entry)[:200])
            continue
        if video_id in seen_video_ids:
            duplicate_video_ids.append(video_id)
        seen_video_ids.add(video_id)

        qa_count_distribution[len(qa_pairs)] += 1
        seen_qtypes: set[str] = set()
        filtered_qtypes: set[str] = set()
        for qa in qa_pairs:
            question_type = str(qa.get("question_type", "")).strip()
            question = str(qa.get("question", "")).strip()
            answer = str(qa.get("answer", "")).strip()
            if not question_type or not question or not answer:
                empty_fields.append(f"video_id={video_id} qa={qa}")
                continue
            if question_type in seen_qtypes:
                raise RuntimeError(
                    f"동일 video_id 안에 중복 question_type이 있습니다: video_id={video_id} question_type={question_type}"
                )
            seen_qtypes.add(question_type)
            if question_type in (config.question_types or []):
                filtered_qtypes.add(question_type)
            elif config.question_types is None and question_type in {
                "accident_place",
                "accident_place_feature",
                "vehicle_a_progress",
                "vehicle_b_progress",
                "fault_ratio",
                "fault_compare",
            }:
                filtered_qtypes.add(question_type)
            elif question_type not in {
                "accident_place",
                "accident_place_feature",
                "vehicle_a_progress",
                "vehicle_b_progress",
                "fault_ratio",
                "fault_compare",
            }:
                unexpected_question_types.append(question_type)

        expected_qtypes = set(config.question_types) if config.question_types else {
            "accident_place",
            "accident_place_feature",
            "vehicle_a_progress",
            "vehicle_b_progress",
            "fault_ratio",
            "fault_compare",
        }
        if filtered_qtypes != expected_qtypes:
            missing_required_qtypes.append(
                f"video_id={video_id} missing={sorted(expected_qtypes - filtered_qtypes)} extra={sorted(filtered_qtypes - expected_qtypes)}"
            )

    print(f"  Raw 비디오 : {len(raw_mp4s)}개")
    print(f"  Label JSON : {len(label_jsons)}개")
    print(f"  QA 항목    : {len(qa_data)}개")
    print(f"  QA 쌍 총계 : {sum(len(entry.get('qa_pairs', [])) for entry in qa_data if isinstance(entry, dict))}개")
    print(f"  3방향 매칭 : {len(matched)}개")
    print(f"  QA 수 분포 : {dict(sorted(qa_count_distribution.items()))}")

    if invalid_entries:
        raise RuntimeError("QA entry 구조가 올바르지 않은 항목이 있습니다.\n" + "\n".join(invalid_entries[:5]))
    if duplicate_video_ids:
        raise RuntimeError("QA dataset에 중복 video_id가 있습니다.\n" + "\n".join(duplicate_video_ids[:5]))
    if empty_fields:
        raise RuntimeError("question/answer/question_type이 비어 있는 QA가 있습니다.\n" + "\n".join(empty_fields[:5]))
    if len(matched) == 0:
        raise RuntimeError("3방향 매칭 결과가 0개입니다. 경로 또는 URL 설정을 확인하세요.")

    if missing_required_qtypes:
        print("  ⚠ 경고: 기대한 question_type 6종이 모두 없는 video가 있습니다.")
        for msg in missing_required_qtypes[:5]:
            print(f"    - {msg}")
    if unexpected_question_types:
        print(
            "  ⚠ 경고: 학습 대상 6종 밖의 question_type이 포함되어 있습니다: "
            f"{sorted(set(unexpected_question_types))[:10]}"
        )



def _verify_label_masking(train_ds: TrafficAccidentQADataset, processor) -> None:
    sample = train_ds[0]
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    answer_mask = labels != -100

    print(f"  전체 시퀀스 : {len(input_ids)} 토큰")
    print(f"  마스킹 구간 : {(~answer_mask).sum().item()} 토큰 (-100)")
    print(f"  답변 구간   : {answer_mask.sum().item()} 토큰")

    if answer_mask.sum().item() == 0:
        print("  ⚠ 경고: 답변 구간이 0입니다. 데이터셋 전처리를 확인하세요.")
        return

    decoded = processor.tokenizer.decode(input_ids[answer_mask], skip_special_tokens=True)
    print(f"  디코딩 답변 : {decoded!r}")


def _enable_runtime_fastpaths() -> None:
    if not torch.cuda.is_available():
        return

    # [수정] A100 계열에서 TF32를 허용하면 matmul 성능이 유의미하게 개선됩니다.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _auto_configure_for_gpu(config: TrainingConfig) -> None:
    """GPU 환경에 맞게 정밀도/체크포인팅만 자동 조정합니다.

    [수정]
    - A100이어도 fps/max_pixels를 자동으로 키우지 않습니다.
      이번 목표는 '정확도 상한'보다 '독립 QA 형태 유지 + 시간 단축'이기 때문입니다.
    """
    if config.no_auto_gpu or not torch.cuda.is_available():
        return

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU  : {gpu_name}")
    print(f"VRAM : {vram_gb:.1f} GB")

    if vram_gb >= 35:
        config.bf16 = True
        config.fp16 = False
        config.gradient_checkpointing = False
        print("→ 고VRAM GPU 감지: bf16=True, gradient_checkpointing=False")
    else:
        config.bf16 = False
        config.fp16 = True
        config.gradient_checkpointing = True
        config.max_pixels = min(config.max_pixels, 320 * 320)
        config.fps = min(config.fps, 0.5)
        print("→ 중간/저VRAM GPU 감지: fp16=True, gradient_checkpointing=True, fps/max_pixels 보수적 유지")


def _run_smoke_test(config: TrainingConfig, train_ds: TrafficAccidentQADataset) -> None:
    print("  (별도 임시 모델로 2 스텝 실행 중...)")
    smoke_base, _ = load_model(
        config.model_id,
        prefer_flash_attention=config.prefer_flash_attention,
    )
    smoke_base.config.use_cache = False
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _finalize_checkpoint(final_dir: Path) -> None:
    from safetensors.torch import load_file

    print("\n[체크포인트 파일 목록]")
    for file_path in sorted(final_dir.iterdir()):
        print(f"  {file_path.name}  ({file_path.stat().st_size / 1e6:.1f} MB)")

    safetensors_path = final_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        weights = load_file(str(safetensors_path))
        best_pt = final_dir.parent / "best.pt"
        torch.save(weights, str(best_pt))
        print(f"\n  best.pt 저장: {best_pt} ({best_pt.stat().st_size / 1e6:.1f} MB, {len(weights)}개 파라미터)")


def _write_evaluation_outputs(
    config: TrainingConfig,
    aggregated: dict,
    per_sample_results: list[dict],
    *,
    train_ds: TrafficAccidentQADataset,
    val_ds: TrafficAccidentQADataset,
    test_ds: TrafficAccidentQADataset,
) -> Path:
    experiment_name = config.experiment_name or Path(config.checkpoint_dir).name
    result_dir = Path(config.results_dir) / experiment_name
    result_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "experiment": experiment_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "model_id": config.model_id,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": list(config.target_modules),
            "learning_rate": config.learning_rate,
            "num_train_epochs": config.num_train_epochs,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "fps": config.fps,
            "max_pixels": config.max_pixels,
            "max_seq_len": config.max_seq_len,
            "gradient_checkpointing": config.gradient_checkpointing,
            "group_samples_by_video": config.group_samples_by_video,
            "video_cache_dir": config.video_cache_dir or "",
        },
        "dataset": {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "train_videos": train_ds.get_distribution()["unique_videos"],
            "val_videos": val_ds.get_distribution()["unique_videos"],
            "test_videos": test_ds.get_distribution()["unique_videos"],
        },
        "metrics": aggregated,
    }

    with open(result_dir / "summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary_payload, file_obj, ensure_ascii=False, indent=2)
    with open(result_dir / "per_sample.json", "w", encoding="utf-8") as file_obj:
        json.dump(per_sample_results, file_obj, ensure_ascii=False, indent=2)

    print(f"[평가 결과 저장] {result_dir}")
    return result_dir


def _print_config(config: TrainingConfig) -> None:
    print("=" * 60)
    print("Qwen3-VL-4B LoRA 미세조정 설정")
    print("-" * 60)
    print(f"  모델              : {config.model_id}")
    print(f"  LoRA rank / alpha : {config.lora_r} / {config.lora_alpha}")
    print(f"  target_modules    : {', '.join(config.target_modules)}")
    print(f"  학습률            : {config.learning_rate}")
    print(f"  에폭              : {config.num_train_epochs}")
    print(f"  배치 크기         : {config.per_device_train_batch_size} × {config.gradient_accumulation_steps}")
    print(f"  fps / max_pixels  : {config.fps} / {config.max_pixels}")
    print(f"  max_seq_len       : {config.max_seq_len}")
    print(f"  bf16 / fp16       : {config.bf16} / {config.fp16}")
    print(f"  grad ckpt         : {config.gradient_checkpointing}")
    print(f"  video grouping    : {config.group_samples_by_video}")
    print(f"  dataloader worker : {config.dataloader_num_workers}")
    print(f"  eval BERTScore    : {config.compute_bertscore}")
    print(f"  비디오 캐시       : {config.video_cache_dir or '(비활성)'}")
    print(f"  체크포인트        : {config.checkpoint_dir}")
    print(f"  결과 디렉터리     : {config.results_dir}")
    if config.raw_url:
        print(f"  RAW URL           : {config.raw_url}")
    if config.label_url:
        print(f"  LABEL URL         : {config.label_url}")
    if config.qa_json_url:
        print(f"  QA URL            : {config.qa_json_url}")
    gpu_info = (
        f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB)"
        if torch.cuda.is_available() else "CPU"
    )
    print(f"  GPU               : {gpu_info}")
    print("=" * 60)


def run_training(config: TrainingConfig | None = None) -> None:
    if config is None:
        config = TrainingConfig()

    _enable_runtime_fastpaths()
    _auto_configure_for_gpu(config)
    config = _prepare_local_data(config)
    _print_config(config)

    print("\n[0/8] 데이터 파일 검증 중...")
    _validate_data_files(config)

    print("\n[1/8] 모델 로드 중...")
    base_model, processor = load_model(
        config.model_id,
        prefer_flash_attention=config.prefer_flash_attention,
    )

    # [수정] train 중 KV cache는 불필요하므로 비활성화해 메모리를 아낍니다.
    base_model.config.use_cache = False

    if config.gradient_checkpointing:
        print("[2/8] Gradient Checkpointing 활성화...")
        # [수정] checkpointing을 쓸 때는 반드시 LoRA 적용 전에 호출해야 gradient 전파가 끊기지 않습니다.
        base_model.enable_input_require_grads()

    print("[3/8] LoRA 적용 중...")
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

    print("\n[4/8] 데이터셋 구성 중...")
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
    for category, count in sorted(dist["by_category"].items()):
        print(f"  {category:<30} {count:>4}샘플")
    print(f"  {'합계':<30} {dist['total']:>4}샘플 ({dist['unique_videos']}개 비디오)")

    print("\n[4.5/8] 라벨 마스킹 검증 중...")
    _verify_label_masking(train_ds, processor)

    if config.precache_train_videos and config.video_cache_dir:
        print("\n[4.6/8] 학습용 비디오 캐시 워밍업...")
        train_ds.warm_video_cache(desc="train")
    if config.precache_eval_videos and config.video_cache_dir:
        print("\n[4.7/8] 평가용 비디오 캐시 워밍업...")
        val_ds.warm_video_cache(desc="val")
        test_ds.warm_video_cache(desc="test")

    print("\n[5/8] TrainingArguments 구성 중...")
    checkpoint_path = Path(config.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # [수정] CUDA에서는 fused AdamW를 사용해 optimizer step 비용을 줄입니다.
    optim_name = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"

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
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=config.report_to,
        run_name=config.run_name,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_prefetch_factor=(
            config.dataloader_prefetch_factor if config.dataloader_num_workers > 0 else None
        ),
        max_steps=config.max_steps,
        remove_unused_columns=False,
        label_names=["labels"],
        prediction_loss_only=True,
        seed=config.seed,
        optim=optim_name,
        max_grad_norm=config.max_grad_norm,
        save_safetensors=True,
    )

    if config.smoke_test:
        print("\n[5.5/8] Smoke test 실행 중...")
        _run_smoke_test(config, train_ds)

    callbacks: list[TrainerCallback] = [_GradDiagCallback()]
    if config.eval_strategy != "no":
        # [수정] epoch 단위 평가 + early stopping으로 불필요한 장기 학습을 줄입니다.
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            )
        )

    print("\n[6/8] 학습 시작...")
    trainer = VideoAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=TrafficAccidentQADataset.collate_fn,
        callbacks=callbacks,
        group_samples_by_video=config.group_samples_by_video,
    )
    trainer.train()

    print("\n[7/8] 어댑터 저장 중...")
    final_dir = checkpoint_path / "final_adapter"
    trainer.model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"  저장 완료: {final_dir}")
    _finalize_checkpoint(final_dir)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[8/8] 최종 평가 실행 중...")
    trainer.model.eval()
    trainer.model.config.use_cache = True

    # [수정] 최종 detailed evaluation은 trainer.predict(logits) 대신 generate 기반 evaluator를 사용합니다.
    eval_config = EvaluatorConfig(
        max_new_tokens=config.generation_max_new_tokens,
        fps=config.fps,
        max_pixels=config.max_pixels,
        max_seq_len=config.max_seq_len,
        system_prompt=config.system_prompt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
        video_cache_dir=config.video_cache_dir,
        reuse_video_cache=bool(config.video_cache_dir),
        sort_by_video=True,
        compute_bertscore=config.compute_bertscore,
        bertscore_batch_size=config.bertscore_batch_size,
    )
    aggregated, per_sample_results = run_evaluation(
        trainer.model,
        processor,
        test_ds.get_active_samples(),
        config=eval_config,
        experiment_name=config.experiment_name or Path(config.checkpoint_dir).name,
    )

    _write_evaluation_outputs(
        config,
        aggregated,
        per_sample_results,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
    )
    print("\n========== 학습 및 평가 완료 ==========")


def _parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Qwen3-VL-4B 교통사고 QA LoRA 미세조정")
    defaults = TrainingConfig()

    parser.add_argument("--qa-json", type=str, default=defaults.qa_json_path)
    parser.add_argument("--video-dir", type=str, default=defaults.raw_video_root)
    parser.add_argument("--label-dir", type=str, default=defaults.label_root)
    parser.add_argument("--output-dir", type=str, default=defaults.checkpoint_dir)
    parser.add_argument("--results-dir", type=str, default=defaults.results_dir)
    parser.add_argument("--download-root", type=str, default=defaults.download_root)
    parser.add_argument("--raw-url", type=str, default=defaults.raw_url)
    parser.add_argument("--label-url", type=str, default=defaults.label_url)
    parser.add_argument("--qa-json-url", type=str, default=defaults.qa_json_url)
    parser.add_argument("--force-download", action="store_true")

    parser.add_argument("--model-id", type=str, default=defaults.model_id)
    parser.add_argument("--no-flash-attn", action="store_true")

    parser.add_argument("--lora-r", type=int, default=defaults.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=defaults.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=defaults.lora_dropout)
    parser.add_argument("--epochs", type=int, default=defaults.num_train_epochs)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--batch-size", type=int, default=defaults.per_device_train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=defaults.per_device_eval_batch_size)
    parser.add_argument("--grad-accum", type=int, default=defaults.gradient_accumulation_steps)
    parser.add_argument("--fps", type=float, default=defaults.fps)
    parser.add_argument("--max-pixels", type=int, default=defaults.max_pixels)
    parser.add_argument("--max-seq-len", type=int, default=defaults.max_seq_len)
    parser.add_argument("--video-cache-dir", type=str, default=defaults.video_cache_dir)
    parser.add_argument("--precache-train-videos", action="store_true")
    parser.add_argument("--precache-eval-videos", action="store_true")
    parser.add_argument("--no-group-by-video", action="store_true")

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--report-to", type=str, default=defaults.report_to, choices=["none", "wandb", "tensorboard"])
    parser.add_argument("--workers", type=int, default=defaults.dataloader_num_workers)
    parser.add_argument("--prefetch-factor", type=int, default=defaults.dataloader_prefetch_factor)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-auto-gpu", action="store_true")
    parser.add_argument("--experiment-name", type=str, default=defaults.experiment_name)
    parser.add_argument("--max-samples-per-category", type=int, default=defaults.max_samples_per_category)
    parser.add_argument("--generation-max-new-tokens", type=int, default=defaults.generation_max_new_tokens)
    parser.add_argument("--compute-bertscore", action="store_true")

    args = parser.parse_args()

    return TrainingConfig(
        qa_json_path=args.qa_json,
        raw_video_root=args.video_dir,
        label_root=args.label_dir,
        checkpoint_dir=args.output_dir,
        results_dir=args.results_dir,
        download_root=args.download_root,
        raw_url=args.raw_url,
        label_url=args.label_url,
        qa_json_url=args.qa_json_url,
        force_download=args.force_download,
        model_id=args.model_id,
        prefer_flash_attention=not args.no_flash_attn,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        fps=args.fps,
        max_pixels=args.max_pixels,
        max_seq_len=args.max_seq_len,
        video_cache_dir=args.video_cache_dir,
        precache_train_videos=args.precache_train_videos,
        precache_eval_videos=args.precache_eval_videos,
        group_samples_by_video=not args.no_group_by_video,
        bf16=not args.no_bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        report_to=args.report_to,
        dataloader_num_workers=args.workers,
        dataloader_prefetch_factor=args.prefetch_factor,
        max_steps=2 if args.smoke_test else args.max_steps,
        no_auto_gpu=args.no_auto_gpu,
        smoke_test=args.smoke_test,
        experiment_name=args.experiment_name,
        max_samples_per_category=args.max_samples_per_category,
        generation_max_new_tokens=args.generation_max_new_tokens,
        compute_bertscore=args.compute_bertscore,
        bertscore_batch_size=defaults.bertscore_batch_size,
    )


def main() -> None:
    config = _parse_args()
    run_training(config)


if __name__ == "__main__":
    main()
