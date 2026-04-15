from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    import sys
    raise ImportError(
        f"qwen_vl_utils를 찾을 수 없습니다.\n"
        f"현재 Python: {sys.executable}\n"
        f"해결: {sys.executable} -m pip install qwen-vl-utils av"
    ) from e


QUESTION_TYPES: list[str] = [
    "accident_place",
    "accident_place_feature",
    "vehicle_a_progress",
    "vehicle_b_progress",
    "fault_ratio",
    "fault_compare",
]


@dataclass
class VideoQASample:
    # 단일 데이터 샘플 (= QA 쌍) 정보
    video_path: Path
    label_path: Path
    question: str
    answer: str
    question_type: str
    video_id: str
    category: str


class TrafficAccidentQADataset(Dataset):
    """교통사고 QA 데이터셋. video_id 단위로 raw/label 매칭 후 question_types 필터링하여 샘플 리스트 구축.
    __getitem__에서 Qwen3-VL 입력 양식에 맞게 텍스트와 시각 정보 처리 후 텐서 반환.
    """

    # 모델에게 부여할 시스템 프롬프트. 모든 대화의 시작점에 들어간다
    DEFAULT_SYSTEM_PROMPT = (
        "당신은 교통사고 영상을 분석하는 전문 분석가입니다. "
        "제공된 블랙박스 영상을 주의 깊게 관찰하고, "
        "사고 상황에 대한 질문에 정확하고 간결하게 답변하세요."
    )

    def __init__(
        self,
        qa_json_path: str | Path,
        raw_video_root: str | Path,
        label_root: str | Path,
        processor: AutoProcessor,                # HuggingFace의 텍스트+비전 통합 프로세서 (토크나이저 역할 포함)
        *,
        fps: float = 1.0,                        # 비디오에서 1초당 추출할 프레임 수 (메모리 관리 목적)
        max_pixels: int = 360 * 420,             # 입력 비디오 프레임의 최대 픽셀 수 (해상도 제한)
        max_seq_len: int = 512,                  # 텍스트 토큰의 최대 길이 (OOM 방지)
        question_types: list[str] | None = None, # 사용할 질문 타입 목록 (None이면 전체 사용)
        indices: list[int] | None = None,        # 전체 샘플 중 사용할 인덱스 목록 (None이면 전체 사용)
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_samples_per_category: int | None = None,  # Phase 1 파일럿: 카테고리별 최대 샘플 수
    ) -> None:
        self.qa_json_path = Path(qa_json_path)
        self.raw_video_root = Path(raw_video_root)
        self.label_root = Path(label_root)
        self.processor = processor
        self.fps = fps
        self.max_pixels = max_pixels
        self.max_seq_len = max_seq_len
        self.question_types = set(question_types or QUESTION_TYPES)
        self.system_prompt = system_prompt
        self.max_samples_per_category = max_samples_per_category

        # 1. 파일 시스템을 뒤져서 (QA, Video, Label) 3가지가 모두 일치하는 유효한 전체 샘플을 구성한다.
        self._all_samples: list[VideoQASample] = self._build_index()

        # 2. indices가 주어졌다면(즉, Train/Val로 쪼개는 상황이라면) 해당 인덱스만 잘라내어 현재 데이터셋으로 씁니다.
        self._samples = (
            [self._all_samples[i] for i in indices]
            if indices is not None
            else self._all_samples
        )
        print(f"[Dataset] 활성 샘플 수: {len(self._samples)} (전체 매칭: {len(self._all_samples)})")

    def _build_index(self) -> list[VideoQASample]:
        """qa_dataset.json × raw/*.mp4 × label/*.json 3방향 매칭 후 flat list 반환."""
        raw_map: dict[str, Path] = {p.stem: p for p in self.raw_video_root.rglob("*.mp4")}
        label_map: dict[str, tuple[Path, str]] = {
            p.stem: (p, p.parent.name) for p in self.label_root.rglob("*.json")
        }

        with open(self.qa_json_path, encoding="utf-8") as f:
            qa_data: list[dict] = json.load(f)

        matched, unmatched, samples = 0, [], []

        # QA 데이터셋을 순회하며 실제 파일이 있는지 확인 후 샘플 리스트에 추가
        for entry in qa_data:
            stem = Path(entry["video_id"]).stem

            # 매칭 실패 케이스 로깅 (QA는 있지만 raw/label이 없는 경우)
            if stem not in raw_map:
                unmatched.append(f"[raw 없음] {entry['video_id']}")
                continue
            if stem not in label_map:
                unmatched.append(f"[label 없음] {entry['video_id']}")
                continue

            video_path = raw_map[stem]
            label_path, category = label_map[stem]
            matched += 1

            # question_types 필터링 후 샘플 생성
            for qa in entry["qa_pairs"]:
                if qa["question_type"] not in self.question_types:
                    continue
                samples.append(VideoQASample(
                    video_path=video_path, label_path=label_path,
                    question=qa["question"], answer=qa["answer"],
                    question_type=qa["question_type"],
                    video_id=entry["video_id"], category=category,
                ))

        # max_samples_per_category: 카테고리(case_code)별 샘플 수 상한 (Phase 1 파일럿용)
        if self.max_samples_per_category is not None:
            samples = _cap_samples_per_category(samples, self.max_samples_per_category)

        print(
            f"[Dataset._build_index] QA: {len(qa_data)} | "
            f"매칭: {matched} | 미매칭: {len(unmatched)} | 샘플: {len(samples)}"
            + (f" (카테고리별 최대 {self.max_samples_per_category}개 적용)" if self.max_samples_per_category else "")
        )

        # 매칭 실패한 케이스가 있다면 최대 5개까지 예시를 보여준다 (디버깅)
        for msg in unmatched[:5]:
            print(f"  {msg}")
        return samples

    def get_distribution(self) -> dict:
        """현재 데이터셋의 카테고리별, 질문 유형별 분포 통계를 반환."""
        return {
            "total": len(self._samples),
            "unique_videos": len({s.video_id for s in self._samples}),
            "by_category": dict(Counter(s.category for s in self._samples)),
            "by_question_type": dict(Counter(s.question_type for s in self._samples)),
        }

    def get_all_samples(self) -> list[VideoQASample]:
        return self._all_samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Qwen3-VL 입력 양식에 맞게 텍스트와 시각 정보 처리 후 텐서 반환.
        - messages: 시스템 프롬프트 + 사용자 질문(텍스트+비디오) + 모델 답변으로 구성된 대화 형식
        - processor.apply_chat_template: Qwen3-VL이 요구하는 텍스트 포맷으로 변환 (토크나이저 역할 포함)
        - process_vision_info: 메시지에서 비디오 정보 추출하여 모델 입력에 맞는 텐서로 변환
        - SFT 레이블 마스킹: 프롬프트 구간은 -100으로 마스킹하여 CrossEntropyLoss 계산에서 제외 (답변 구간만 학습 대상)
        """
        sample = self._samples[idx]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video", 
                        "video": str(sample.video_path),
                        "fps": self.fps, 
                        "max_pixels": self.max_pixels
                    },
                    {
                        "type": "text", 
                        "text": sample.question
                    },
                ],
            },
            {"role": "assistant", "content": sample.answer},
        ]

        # 1. full_text: System + User(질문) + Assistant(정답)이 모두 포함된 전체 텍스트입니다. (학습에 사용)
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # 2. prompt_text: System + User(질문)까지만 포함된 텍스트입니다.
        # 모델이 답변을 생성하기 시작하는 시점까지의 프롬프트로, SFT 레이블 마스킹에서 -100으로 처리되어 학습에서 제외됩니다.
        prompt_text = self.processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )

        # 3. 시각 정보 처리: 메시지에서 비디오 정보를 추출하여 모델 입력에 맞는 텐서로 변환합니다.
        image_inputs, video_inputs = process_vision_info(messages)

        # 4. processor를 사용하여 텍스트와 시각 정보를 모델 입력 양식에 맞게 텐서로 변환합니다.
        full_batch = self.processor(
            text=[full_text], images=image_inputs, videos=video_inputs,
            return_tensors="pt", truncation=True, max_length=self.max_seq_len,
        )
        # 프롬프트(질문)까지만 별도로 텐서로 변환하여 길이를 계산합니다. 답변 부분은 모델이 생성하는 시점부터 시작하므로, 프롬프트 길이만큼 레이블을 -100으로 마스킹하여 학습에서 제외합니다.
        prompt_batch = self.processor(
            text=[prompt_text], images=image_inputs, videos=video_inputs,
            return_tensors="pt", truncation=True, max_length=self.max_seq_len,
        )

        # full_batch의 input_ids에서 prompt_batch의 길이만큼 앞부분을 -100으로 마스킹하여 레이블로 사용합니다. 이렇게 하면 모델이 답변을 생성하는 시점부터 학습이 시작되고, 프롬프트 부분은 손실 계산에서 제외됩니다.
        full_ids = full_batch["input_ids"][0]
        prompt_len = prompt_batch["input_ids"][0].shape[0]

        if prompt_len >= full_ids.shape[0]:
            print(
                f"[경고] 답변이 잘렸습니다 (prompt_len={prompt_len} >= full_len={full_ids.shape[0]}). "
                f"max_seq_len 증가 필요. video_id={sample.video_id}"
            )

        # SFT 레이블 마스킹: 프롬프트 구간은 -100 → CrossEntropyLoss에서 무시됨
        labels = full_ids.clone()
        labels[:prompt_len] = -100

        result: dict[str, torch.Tensor] = {
            "input_ids": full_ids,
            "attention_mask": full_batch["attention_mask"][0],
            "labels": labels,
        }
        # 비전 모델이 사용하는 특수 텐서들(픽셀값, 그리드 정보)이 생성되었다면 함께 넘겨줍니다.
        for key in ["pixel_values", "image_grid_thw", "video_grid_thw"]:
            if key in full_batch:
                result[key] = full_batch[key]
        return result

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        """
        배치 내 샘플들의 텐서들을 모델 입력 양식에 맞게 패딩하여 하나의 텐서로 묶어줍니다.
        - input_ids, attention_mask, labels는 시퀀스 길이가 다른 샘플들을 패딩하여 최대 길이에 맞춥니다. (Qwen3-VL은 left-padding 사용)
        - pixel_values, image_grid_thw, video_grid_thw는 시각 텐서로, 배치 내 샘플들을 단순히 concat하여 하나의 텐서로 만듭니다. (Qwen3-VL이 배치 내 시각 텐서를 concat하여 처리하기 때문)
        - 결과적으로 input_ids, attention_mask, labels는 (batch_size, max_seq_len) 형태의 텐서가 되고, 시각 텐서들은 (total_frames, C, H, W) 또는 (total_frames, grid_h, grid_w) 형태의 텐서가 됩니다.
        """ 
        max_len = max(item["input_ids"].shape[0] for item in batch) # 가장 긴문장
        padded_ids, padded_mask, padded_labels = [], [], []

        for item in batch:
            # 가장 긴 문장과의 길이 차이만큼 왼쪽에 채워넣을 패딩 길이를 계산합니다.
            pad_len = max_len - item["input_ids"].shape[0]
            padded_ids.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), item["input_ids"]]))
            padded_mask.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), item["attention_mask"]]))
            padded_labels.append(torch.cat([torch.full((pad_len,), -100, dtype=torch.long), item["labels"]]))

        result: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(padded_mask),
            "labels": torch.stack(padded_labels),
        }
        for key in ["pixel_values", "image_grid_thw", "video_grid_thw"]:
            tensors = [item[key] for item in batch if key in item]
            if tensors:
                result[key] = torch.cat(tensors, dim=0)
        return result


def _cap_samples_per_category(
    samples: list[VideoQASample], max_per_category: int
) -> list[VideoQASample]:
    """카테고리(case_code)별로 video_id 단위로 최대 N개 샘플만 남깁니다.

    Phase 1 파일럿(case_code별 10개)과 Phase 2 soft capping(200개)에 사용됩니다.
    video_id 단위로 제한해 데이터 릭을 방지합니다.
    """
    from collections import defaultdict

    # category → video_id 집합
    cat_video_ids: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        if s.video_id not in cat_video_ids[s.category]:
            cat_video_ids[s.category].append(s.video_id)

    # 카테고리별 허용 video_id (최대 max_per_category개)
    allowed_ids: set[str] = set()
    for cat, video_ids in cat_video_ids.items():
        allowed_ids.update(video_ids[:max_per_category])

    return [s for s in samples if s.video_id in allowed_ids]


def stratified_split_by_category(
    samples: list[VideoQASample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[set[str], set[str], set[str]]:
    """카테고리(case_code)별 계층적 train/val/test 분할.

    전체 video_id를 무작위 셔플한 뒤 나누는 기존 방식은 운이 나쁘면 특정 카테고리가
    test에만 몰리는 클래스 불균형이 생깁니다.
    이 함수는 각 카테고리 안에서 독립적으로 비율 분할하여 모든 split에
    카테고리가 균등하게 포함되도록 보장합니다.

    Args:
        samples:     VideoQASample 전체 리스트 (video_id·category 속성만 사용)
        train_ratio: 학습 비율 (기본 0.7)
        val_ratio:   검증 비율 (기본 0.2)
        seed:        재현성 시드

    Returns:
        (train_ids, val_ids, test_ids) — 각각 video_id 문자열 집합

    Edge-case:
        카테고리당 비디오 수가 적을 때:
          n=1 → train만,  n=2 → train 1 + test 1,  n=3 → train 2 + test 1
    """
    # category별 unique video_id 수집 (정렬로 시드 고정 시 재현성 보장)
    cat_to_ids: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for s in samples:
        if s.video_id not in seen:
            seen.add(s.video_id)
            cat_to_ids[s.category].append(s.video_id)

    rng = random.Random(seed)

    train_ids: set[str] = set()
    val_ids:   set[str] = set()
    test_ids:  set[str] = set()

    for cat, video_ids in sorted(cat_to_ids.items()):
        ids = sorted(video_ids)   # 정렬 후 셔플 → 시드 고정 시 재현성 보장
        rng.shuffle(ids)

        n       = len(ids)
        n_train = max(1, int(round(n * train_ratio)))
        n_val   = int(round(n * val_ratio))

        # 합이 전체를 초과하지 않도록 클램프
        n_train = min(n_train, n)
        n_val   = min(n_val,   n - n_train)
        # test는 나머지 전부 (반올림 오차 누적 방지)

        train_ids.update(ids[:n_train])
        val_ids.update(ids[n_train : n_train + n_val])
        test_ids.update(ids[n_train + n_val :])

    return train_ids, val_ids, test_ids


def _log_split_stats(
    all_samples: list[VideoQASample],
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
) -> None:
    """카테고리별 분할 분포를 요약 출력합니다."""
    cat_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for s in all_samples:
        if s.video_id in train_ids:
            cat_counts[s.category]["train"] += 1
        elif s.video_id in val_ids:
            cat_counts[s.category]["val"] += 1
        elif s.video_id in test_ids:
            cat_counts[s.category]["test"] += 1

    cats_missing_test = [c for c, v in cat_counts.items() if v["test"] == 0]
    cats_missing_val  = [c for c, v in cat_counts.items() if v["val"] == 0]
    print(
        f"[분할 통계] 카테고리 수: {len(cat_counts)} | "
        f"test 비어있는 카테고리: {len(cats_missing_test)} | "
        f"val 비어있는 카테고리: {len(cats_missing_val)}"
    )
    if cats_missing_test:
        print(f"  → test 없음 (데이터 부족): {cats_missing_test[:5]}"
              + (" ..." if len(cats_missing_test) > 5 else ""))


def build_dataloaders(
    qa_json_path: str | Path,
    raw_video_root: str | Path,
    label_root: str | Path,
    processor: AutoProcessor,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    # test 비율은 1 - train_ratio - val_ratio (기본 0.1)
    seed: int = 42,
    batch_size: int = 1,
    fps: float = 1.0,
    max_pixels: int = 360 * 420,
    max_seq_len: int = 512,
    question_types: list[str] | None = None,
    num_workers: int = 0,
    system_prompt: str = TrafficAccidentQADataset.DEFAULT_SYSTEM_PROMPT,
    max_samples_per_category: int | None = None,
) -> tuple[
    TrafficAccidentQADataset, TrafficAccidentQADataset, TrafficAccidentQADataset,
    DataLoader, DataLoader, DataLoader,
]:
    """video 단위 train/val/test 분할 후 Dataset·DataLoader 3쌍 반환.

    반환 순서: train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
    """
    full_ds = TrafficAccidentQADataset(
        qa_json_path, raw_video_root, label_root, processor,
        fps=fps, max_pixels=max_pixels, max_seq_len=max_seq_len,
        question_types=question_types, system_prompt=system_prompt,
        max_samples_per_category=max_samples_per_category,
    )
    all_samples = full_ds.get_all_samples()

    # [계층적 분할 — case_code별 균등 분할로 클래스 불균형 방지]
    # 카테고리마다 독립적으로 비율 분할하므로, video_id 단위 분리도 자동 보장됩니다.
    train_ids, val_ids, test_ids = stratified_split_by_category(
        all_samples, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    train_idx = [i for i, s in enumerate(all_samples) if s.video_id in train_ids]
    val_idx   = [i for i, s in enumerate(all_samples) if s.video_id in val_ids]
    test_idx  = [i for i, s in enumerate(all_samples) if s.video_id in test_ids]

    print(
        f"[build_dataloaders] "
        f"학습: {len(train_ids)}개 비디오 / {len(train_idx)}샘플 | "
        f"검증: {len(val_ids)}개 비디오 / {len(val_idx)}샘플 | "
        f"테스트: {len(test_ids)}개 비디오 / {len(test_idx)}샘플"
    )
    _log_split_stats(all_samples, train_ids, val_ids, test_ids)

    _kw = dict(
        fps=fps, max_pixels=max_pixels, max_seq_len=max_seq_len,
        question_types=question_types, system_prompt=system_prompt,
        max_samples_per_category=max_samples_per_category,
    )
    train_ds = TrafficAccidentQADataset(qa_json_path, raw_video_root, label_root, processor, indices=train_idx, **_kw)
    val_ds   = TrafficAccidentQADataset(qa_json_path, raw_video_root, label_root, processor, indices=val_idx,   **_kw)
    test_ds  = TrafficAccidentQADataset(qa_json_path, raw_video_root, label_root, processor, indices=test_idx,  **_kw)

    _loader_kw = dict(collate_fn=TrafficAccidentQADataset.collate_fn,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **_loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **_loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **_loader_kw)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
