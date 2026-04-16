from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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


def _stable_path_hash(path: Path) -> str:
    """[수정] 같은 stem 충돌을 피하기 위해 경로 기반 해시를 캐시 키에 섞습니다."""
    resolved = path.resolve(strict=False)
    return hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]


def _build_unique_stem_index(paths: list[Path], *, label_with_parent: bool = False) -> dict:
    """[수정] stem 중복을 조용히 덮어쓰지 않고 즉시 실패시켜 잘못된 매핑을 막습니다."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[path.stem].append(path)

    duplicates = {stem: items for stem, items in grouped.items() if len(items) > 1}
    if duplicates:
        preview = []
        for stem, items in list(sorted(duplicates.items()))[:5]:
            preview.append(f"{stem}: {[str(item) for item in items[:3]]}")
        raise RuntimeError(
            "stem 기준 중복 파일이 있어 raw/label 매핑이 모호합니다.\n"
            + "\n".join(preview)
        )

    if label_with_parent:
        return {stem: (items[0], items[0].parent.name) for stem, items in grouped.items()}
    return {stem: items[0] for stem, items in grouped.items()}


@dataclass(slots=True)
class VideoQASample:
    """단일 QA 샘플.

    [수정]
    - per_qa 방식을 유지하되, 같은 비디오를 재사용할 수 있게 프롬프트/토큰 길이/캐시 키를 미리 저장합니다.
    - sample = (video, question, answer) 구조는 그대로 유지하므로 추론 형태와 학습 형태가 일치합니다.
    """

    video_path: Path
    label_path: Path
    question: str
    answer: str
    question_type: str
    video_id: str
    category: str
    full_text: str = ""
    prompt_token_len: int = 0
    # [수정] partial truncation 감지를 위해 answer token 길이를 함께 저장합니다.
    answer_token_len: int = 0
    video_cache_key: str = ""


class TrafficAccidentQADataset(Dataset):
    """교통사고 QA 데이터셋.

    [수정 핵심]
    1) 학습 단위는 계속 per_qa(독립 질문 1개)로 유지
    2) raw/label/qa 인덱스는 1회만 생성하고 split dataset은 view만 재사용
    3) 같은 영상은 프레임 캐시(.npy + 메모리 cache)로 재디코딩 비용 최소화
    4) prompt 길이 계산은 텍스트 토크나이징 결과를 재사용해 비전 전처리 중복 제거
    """

    DEFAULT_SYSTEM_PROMPT = (
        "당신은 교통사고 영상을 분석하는 전문 분석가입니다. "
        "제공된 블랙박스 영상을 주의 깊게 관찰하고, "
        "사고 상황에 대한 질문에 정확하고 간결하게 답변하세요."
    )

    _PAD_TOKEN_ID: int = 0

    def __init__(
        self,
        qa_json_path: str | Path,
        raw_video_root: str | Path,
        label_root: str | Path,
        processor: AutoProcessor,
        *,
        fps: float = 1.0,
        max_pixels: int = 360 * 420,
        max_seq_len: int = 512,
        question_types: list[str] | None = None,
        indices: list[int] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_samples_per_category: int | None = None,
        video_cache_dir: str | Path | None = None,
        shared_all_samples: list[VideoQASample] | None = None,
        shared_frame_cache: dict[str, list] | None = None,
        shared_visual_feature_cache: dict[str, dict[str, torch.Tensor]] | None = None,
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

        # [수정] pad token을 tokenizer 기준으로 안전하게 맞춰 left-padding 시 오류를 막습니다.
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", 0)
        self.__class__._PAD_TOKEN_ID = pad_token_id if pad_token_id is not None else eos_token_id

        # [수정] 디스크 캐시 + 메모리 캐시를 동시에 사용합니다.
        self._cache_dir: Path | None = Path(video_cache_dir) if video_cache_dir else None
        self._frame_cache: dict[str, list] = shared_frame_cache if shared_frame_cache is not None else {}
        # [수정] 같은 비디오의 pixel_values / video_grid_thw는 질문과 무관하므로 한 번만 계산해 재사용합니다.
        self._visual_feature_cache: dict[str, dict[str, torch.Tensor]] = (
            shared_visual_feature_cache if shared_visual_feature_cache is not None else {}
        )
        # [수정] text-only tokenization 결과와 multimodal processor의 input_ids가 일치하는지 1회 검증합니다.
        self._text_only_visual_merge_supported: bool | None = None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Dataset] 프레임 캐시 활성화: {self._cache_dir}")

        # [수정] 같은 경고를 반복 출력하지 않도록 truncation 경고 키를 기억합니다.
        self._warned_truncation_keys: set[str] = set()

        # [수정] split마다 _build_index()를 다시 돌리지 않도록 전체 인덱스를 공유합니다.
        self._all_samples: list[VideoQASample] = (
            shared_all_samples if shared_all_samples is not None else self._build_index()
        )
        self._samples = [self._all_samples[i] for i in indices] if indices is not None else self._all_samples
        print(f"[Dataset] 활성 샘플 수: {len(self._samples)} (전체 매칭: {len(self._all_samples)})")

    def subset(self, indices: list[int]) -> "TrafficAccidentQADataset":
        """[수정] 전체 인덱스를 재스캔하지 않고 현재 데이터셋의 view만 만듭니다."""
        return TrafficAccidentQADataset(
            qa_json_path=self.qa_json_path,
            raw_video_root=self.raw_video_root,
            label_root=self.label_root,
            processor=self.processor,
            fps=self.fps,
            max_pixels=self.max_pixels,
            max_seq_len=self.max_seq_len,
            question_types=list(self.question_types),
            indices=indices,
            system_prompt=self.system_prompt,
            max_samples_per_category=self.max_samples_per_category,
            video_cache_dir=self._cache_dir,
            shared_all_samples=self._all_samples,
            shared_frame_cache=self._frame_cache,
            shared_visual_feature_cache=self._visual_feature_cache,
        )

    def _resolve_system_prompt(self, category: str) -> str:
        if "{case_code}" in self.system_prompt:
            return self.system_prompt.format(case_code=category)
        return self.system_prompt

    def _build_video_cache_key(self, video_path: Path) -> str:
        # [수정] stem만 쓰면 다른 디렉터리의 동명 파일과 충돌할 수 있어 경로 해시를 함께 사용합니다.
        return f"{video_path.stem}_{_stable_path_hash(video_path)}_fps{self.fps}_px{self.max_pixels}"

    def _token_len(self, text: str) -> int:
        """[수정] prompt 길이 계산용 text-only tokenization 헬퍼."""
        return int(
            self.processor(
                text=[text],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
            )["input_ids"][0].shape[0]
        )

    def _answer_token_len(self, text: str) -> int:
        """[수정] partial truncation 감지를 위해 answer 자체의 토큰 길이를 별도 계산합니다."""
        return int(
            self.processor.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
                add_special_tokens=False,
            )["input_ids"][0].shape[0]
        )

    def _build_index(self) -> list[VideoQASample]:
        """qa_dataset.json × raw/*.mp4 × label/*.json 3방향 매칭 후 per_qa 샘플 생성."""
        # [수정] stem 중복이 있으면 조용히 덮어쓰지 않고 즉시 실패시켜 데이터 매핑 오류를 막습니다.
        raw_map: dict[str, Path] = _build_unique_stem_index(list(self.raw_video_root.rglob("*.mp4")))
        label_map: dict[str, tuple[Path, str]] = _build_unique_stem_index(
            list(self.label_root.rglob("*.json")),
            label_with_parent=True,
        )

        with open(self.qa_json_path, encoding="utf-8") as file_obj:
            qa_data: list[dict] = json.load(file_obj)

        matched, unmatched, samples = 0, [], []
        prompt_len_cache: dict[tuple[str, str, str], int] = {}
        seen_video_ids: set[str] = set()

        for entry in qa_data:
            video_id = str(entry.get("video_id", "")).strip()
            qa_pairs = entry.get("qa_pairs")
            if not video_id or not isinstance(qa_pairs, list):
                raise RuntimeError(f"QA entry 구조가 올바르지 않습니다: {entry}")
            if video_id in seen_video_ids:
                raise RuntimeError(f"QA dataset에 중복 video_id가 있습니다: {video_id}")
            seen_video_ids.add(video_id)

            stem = Path(video_id).stem
            if stem not in raw_map:
                unmatched.append(f"[raw 없음] {video_id}")
                continue
            if stem not in label_map:
                unmatched.append(f"[label 없음] {video_id}")
                continue

            video_path = raw_map[stem]
            label_path, category = label_map[stem]
            effective_system_prompt = self._resolve_system_prompt(category)
            matched += 1

            seen_question_types: set[str] = set()
            for qa in qa_pairs:
                question_type = str(qa.get("question_type", "")).strip()
                question = str(qa.get("question", "")).strip()
                answer = str(qa.get("answer", "")).strip()

                # [수정] question/answer/question_type 누락과 중복 question_type을 즉시 검출합니다.
                if not question_type or not question or not answer:
                    raise RuntimeError(f"QA pair 필수값이 비어 있습니다: video_id={video_id} qa={qa}")
                if question_type in seen_question_types:
                    raise RuntimeError(
                        f"동일 video_id 안에 중복 question_type이 있습니다: video_id={video_id} question_type={question_type}"
                    )
                seen_question_types.add(question_type)

                if question_type not in self.question_types:
                    continue

                prompt_messages = [
                    {"role": "system", "content": effective_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": str(video_path),
                                "fps": self.fps,
                                "max_pixels": self.max_pixels,
                            },
                            {"type": "text", "text": question},
                        ],
                    },
                ]

                # [수정] 동일한 (prompt, video) 조합의 prompt token 길이는 미리 캐시합니다.
                prompt_key = (effective_system_prompt, str(video_path), question)
                if prompt_key not in prompt_len_cache:
                    prompt_text = self.processor.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    prompt_len_cache[prompt_key] = self._token_len(prompt_text)

                full_text = self.processor.apply_chat_template(
                    prompt_messages + [{"role": "assistant", "content": answer}],
                    tokenize=False,
                    add_generation_prompt=False,
                )

                samples.append(
                    VideoQASample(
                        video_path=video_path,
                        label_path=label_path,
                        question=question,
                        answer=answer,
                        question_type=question_type,
                        video_id=video_id,
                        category=category,
                        full_text=full_text,
                        prompt_token_len=prompt_len_cache[prompt_key],
                        answer_token_len=self._answer_token_len(answer),
                        video_cache_key=self._build_video_cache_key(video_path),
                    )
                )

        if self.max_samples_per_category is not None:
            samples = _cap_samples_per_category(samples, self.max_samples_per_category)

        print(
            f"[Dataset._build_index] QA: {len(qa_data)} | "
            f"매칭: {matched} | 미매칭: {len(unmatched)} | 샘플: {len(samples)}"
            + (
                f" (카테고리별 최대 {self.max_samples_per_category}개 적용)"
                if self.max_samples_per_category is not None
                else ""
            )
        )
        for msg in unmatched[:5]:
            print(f"  {msg}")
        return samples

    def get_distribution(self) -> dict:
        return {
            "total": len(self._samples),
            "unique_videos": len({sample.video_id for sample in self._samples}),
            "by_category": dict(Counter(sample.category for sample in self._samples)),
            "by_question_type": dict(Counter(sample.question_type for sample in self._samples)),
        }

    def get_all_samples(self) -> list[VideoQASample]:
        return self._all_samples

    def get_active_samples(self) -> list[VideoQASample]:
        return self._samples

    def grouped_indices_by_video(self) -> list[list[int]]:
        """[수정] train sampler가 같은 video_id 샘플을 붙여 순회할 수 있게 인덱스를 그룹화합니다."""
        grouped: dict[str, list[int]] = defaultdict(list)
        for idx, sample in enumerate(self._samples):
            grouped[sample.video_id].append(idx)
        return list(grouped.values())

    def warm_video_cache(self, *, desc: str = "cache", max_unique_videos: int | None = None) -> None:
        """[수정] 첫 epoch 전에 캐시를 미리 채워 디코딩 병목을 앞당겨 제거합니다."""
        unique_paths: list[Path] = []
        seen: set[str] = set()

        for sample in self._samples:
            path_text = str(sample.video_path)
            if path_text in seen:
                continue
            seen.add(path_text)
            unique_paths.append(sample.video_path)
            if max_unique_videos is not None and len(unique_paths) >= max_unique_videos:
                break

        total = len(unique_paths)
        if total == 0:
            return

        print(f"[Dataset] {desc} 프레임 캐시 워밍업 시작: {total}개 비디오")
        for idx, video_path in enumerate(unique_paths, start=1):
            self._get_video_inputs(str(video_path))
            if idx == total or idx % 50 == 0:
                print(f"  - {idx}/{total}")

    def __len__(self) -> int:
        return len(self._samples)

    def _get_video_inputs(self, video_path: str) -> list:
        """비디오 프레임을 메모리/디스크 캐시에서 불러오거나 새로 디코딩합니다."""
        if video_path in self._frame_cache:
            return self._frame_cache[video_path]

        cache_file: Path | None = None
        if self._cache_dir is not None:
            cache_key = self._build_video_cache_key(Path(video_path))
            cache_file = self._cache_dir / f"{cache_key}.npy"
            if cache_file.exists():
                try:
                    frames = np.load(str(cache_file), allow_pickle=False)
                    video_inputs = [frames]
                    self._frame_cache[video_path] = video_inputs
                    return video_inputs
                except (ValueError, OSError):
                    cache_file.unlink(missing_ok=True)

        _, video_inputs = process_vision_info([
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": self.fps,
                        "max_pixels": self.max_pixels,
                    }
                ],
            }
        ])

        # [수정] atomic rename으로 worker 간 동시 저장 시 파일 깨짐을 방지합니다.
        if cache_file is not None and video_inputs:
            tmp_file = cache_file.with_name(f"{cache_file.stem}.{os.getpid()}.tmp.npy")
            np.save(str(tmp_file), video_inputs[0])
            os.replace(str(tmp_file), str(cache_file))

        self._frame_cache[video_path] = video_inputs
        return video_inputs

    def _build_text_batch(self, text: str) -> dict[str, torch.Tensor]:
        """[수정] 비디오와 무관한 input_ids/attention_mask는 text-only tokenization으로 생성합니다."""
        return self.processor(
            text=[text],
            images=None,
            videos=None,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        )

    @staticmethod
    def _extract_visual_features(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """[수정] 질문과 무관한 비전 텐서만 분리해 비디오 단위로 재사용합니다."""
        visual: dict[str, torch.Tensor] = {}
        for key in ["pixel_values", "image_grid_thw", "video_grid_thw"]:
            if key in batch:
                visual[key] = batch[key].clone()
        return visual

    @staticmethod
    def _find_answer_start_index(full_ids: torch.Tensor, answer_ids: torch.Tensor) -> int | None:
        """[수정] 실제 full input_ids에서 answer suffix 시작 위치를 역탐색해 라벨 마스킹 경계를 정확히 잡습니다."""
        if answer_ids.numel() == 0 or full_ids.numel() < answer_ids.numel():
            return None

        full_list = full_ids.tolist()
        answer_list = answer_ids.tolist()
        answer_len = len(answer_list)

        for start_idx in range(len(full_list) - answer_len, -1, -1):
            if full_list[start_idx:start_idx + answer_len] == answer_list:
                return start_idx
        return None

    def _get_full_batch(self, sample: VideoQASample, video_inputs: list) -> dict[str, torch.Tensor]:
        """[수정] text-only + cached visual merge가 가능하면 같은 비디오의 비전 전처리를 한 번만 사용합니다."""
        visual_cache_key = str(sample.video_path)
        text_batch = self._build_text_batch(sample.full_text)

        cached_visual = self._visual_feature_cache.get(visual_cache_key)
        if cached_visual is not None and self._text_only_visual_merge_supported is True:
            merged_batch = dict(text_batch)
            merged_batch.update(cached_visual)
            return merged_batch

        multimodal_batch = self.processor(
            text=[sample.full_text],
            images=None,
            videos=video_inputs,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        )

        # [수정] multimodal input_ids와 text-only input_ids가 실제로 같을 때만 빠른 경로를 활성화합니다.
        ids_match = torch.equal(multimodal_batch["input_ids"], text_batch["input_ids"])
        mask_match = torch.equal(multimodal_batch["attention_mask"], text_batch["attention_mask"])
        if self._text_only_visual_merge_supported is None:
            self._text_only_visual_merge_supported = bool(ids_match and mask_match)
            print(
                "[Dataset] text-only + visual cache 병합 "
                f"{'활성화' if self._text_only_visual_merge_supported else '비활성화'}"
            )

        if self._text_only_visual_merge_supported and ids_match and mask_match:
            self._visual_feature_cache[visual_cache_key] = self._extract_visual_features(multimodal_batch)
            merged_batch = dict(text_batch)
            merged_batch.update(self._visual_feature_cache[visual_cache_key])
            return merged_batch

        # [수정] 호환되지 않으면 안전하게 기존 multimodal processor 결과를 그대로 사용합니다.
        self._text_only_visual_merge_supported = False
        return multimodal_batch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """per_qa 샘플을 Qwen3-VL 입력 텐서로 변환합니다."""
        sample = self._samples[idx]

        # [수정] 동일 비디오는 캐시에서 프레임을 재사용합니다.
        video_inputs = self._get_video_inputs(str(sample.video_path))
        full_batch = self._get_full_batch(sample, video_inputs)

        full_ids = full_batch["input_ids"][0]
        answer_ids = self.processor.tokenizer(
            sample.answer,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=False,
        )["input_ids"][0]

        # [수정] 실제 answer suffix를 찾아 prompt_len 추정 대신 정답 시작 위치를 직접 계산합니다.
        answer_start_idx = self._find_answer_start_index(full_ids, answer_ids)
        if answer_start_idx is None:
            prompt_len = min(sample.prompt_token_len, int(full_ids.shape[0]))
            warning_key = f"{sample.video_id}:{sample.question_type}:answer_span_fallback"
            if warning_key not in self._warned_truncation_keys:
                self._warned_truncation_keys.add(warning_key)
                print(
                    f"[경고] answer suffix를 찾지 못해 기존 prompt_len fallback을 사용합니다. "
                    f"video_id={sample.video_id} question_type={sample.question_type}"
                )
        else:
            prompt_len = answer_start_idx

        if prompt_len >= full_ids.shape[0]:
            warning_key = f"{sample.video_id}:{sample.question_type}:prompt_full"
            if warning_key not in self._warned_truncation_keys:
                self._warned_truncation_keys.add(warning_key)
                print(
                    f"[경고] 답변이 완전히 잘렸습니다 (prompt_len={prompt_len} >= full_len={full_ids.shape[0]}). "
                    f"max_seq_len 증가 필요. video_id={sample.video_id} question_type={sample.question_type}"
                )

        labels = full_ids.clone()
        labels[:prompt_len] = -100

        answer_token_count = int((labels != -100).sum().item())
        if 0 < answer_token_count < sample.answer_token_len:
            warning_key = f"{sample.video_id}:{sample.question_type}:partial"
            if warning_key not in self._warned_truncation_keys:
                self._warned_truncation_keys.add(warning_key)
                print(
                    f"[경고] 답변이 부분 truncation 되었습니다 "
                    f"(kept={answer_token_count}, expected≈{sample.answer_token_len}). "
                    f"video_id={sample.video_id} question_type={sample.question_type}"
                )

        result: dict[str, torch.Tensor] = {
            "input_ids": full_ids,
            "attention_mask": full_batch["attention_mask"][0],
            "labels": labels,
        }
        for key in ["pixel_values", "image_grid_thw", "video_grid_thw"]:
            if key in full_batch:
                result[key] = full_batch[key]
        return result

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        """left-padding 기준으로 배치를 묶습니다."""
        max_len = max(item["input_ids"].shape[0] for item in batch)
        pad_token_id = TrafficAccidentQADataset._PAD_TOKEN_ID
        padded_ids, padded_mask, padded_labels = [], [], []

        for item in batch:
            pad_len = max_len - item["input_ids"].shape[0]
            padded_ids.append(
                torch.cat([
                    torch.full((pad_len,), pad_token_id, dtype=torch.long),
                    item["input_ids"],
                ])
            )
            padded_mask.append(
                torch.cat([
                    torch.zeros(pad_len, dtype=torch.long),
                    item["attention_mask"],
                ])
            )
            padded_labels.append(
                torch.cat([
                    torch.full((pad_len,), -100, dtype=torch.long),
                    item["labels"],
                ])
            )

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
    """카테고리(case_code)별로 video_id 단위로 최대 N개 비디오만 남깁니다."""
    cat_video_ids: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        if sample.video_id not in cat_video_ids[sample.category]:
            cat_video_ids[sample.category].append(sample.video_id)

    allowed_ids: set[str] = set()
    for _, video_ids in cat_video_ids.items():
        allowed_ids.update(video_ids[:max_per_category])

    return [sample for sample in samples if sample.video_id in allowed_ids]


def stratified_split_by_category(
    samples: list[VideoQASample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[set[str], set[str], set[str]]:
    """카테고리(case_code)별 계층적 train/val/test 분할."""
    cat_to_ids: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for sample in samples:
        if sample.video_id not in seen:
            seen.add(sample.video_id)
            cat_to_ids[sample.category].append(sample.video_id)

    rng = random.Random(seed)
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    test_ids: set[str] = set()

    for _, video_ids in sorted(cat_to_ids.items()):
        ids = sorted(video_ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = int(round(n * val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        train_ids.update(ids[:n_train])
        val_ids.update(ids[n_train:n_train + n_val])
        test_ids.update(ids[n_train + n_val:])

    return train_ids, val_ids, test_ids


def _log_split_stats(
    all_samples: list[VideoQASample],
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
) -> None:
    cat_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for sample in all_samples:
        if sample.video_id in train_ids:
            cat_counts[sample.category]["train"] += 1
        elif sample.video_id in val_ids:
            cat_counts[sample.category]["val"] += 1
        elif sample.video_id in test_ids:
            cat_counts[sample.category]["test"] += 1

    cats_missing_test = [cat for cat, counts in cat_counts.items() if counts["test"] == 0]
    cats_missing_val = [cat for cat, counts in cat_counts.items() if counts["val"] == 0]
    print(
        f"[분할 통계] 카테고리 수: {len(cat_counts)} | "
        f"test 비어있는 카테고리: {len(cats_missing_test)} | "
        f"val 비어있는 카테고리: {len(cats_missing_val)}"
    )
    if cats_missing_test:
        print(
            f"  → test 없음 (데이터 부족): {cats_missing_test[:5]}"
            + (" ..." if len(cats_missing_test) > 5 else "")
        )


def build_dataloaders(
    qa_json_path: str | Path,
    raw_video_root: str | Path,
    label_root: str | Path,
    processor: AutoProcessor,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
    batch_size: int = 1,
    fps: float = 1.0,
    max_pixels: int = 360 * 420,
    max_seq_len: int = 512,
    question_types: list[str] | None = None,
    num_workers: int = 0,
    system_prompt: str = TrafficAccidentQADataset.DEFAULT_SYSTEM_PROMPT,
    max_samples_per_category: int | None = None,
    video_cache_dir: str | Path | None = None,
) -> tuple[
    TrafficAccidentQADataset, TrafficAccidentQADataset, TrafficAccidentQADataset,
    DataLoader, DataLoader, DataLoader,
]:
    """video 단위 train/val/test 분할 후 Dataset·DataLoader 3쌍 반환."""
    # [수정] 전체 index를 1회만 만들고 split view를 재사용합니다.
    full_ds = TrafficAccidentQADataset(
        qa_json_path=qa_json_path,
        raw_video_root=raw_video_root,
        label_root=label_root,
        processor=processor,
        fps=fps,
        max_pixels=max_pixels,
        max_seq_len=max_seq_len,
        question_types=question_types,
        system_prompt=system_prompt,
        max_samples_per_category=max_samples_per_category,
        video_cache_dir=video_cache_dir,
    )
    all_samples = full_ds.get_all_samples()

    train_ids, val_ids, test_ids = stratified_split_by_category(
        all_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_idx = [i for i, sample in enumerate(all_samples) if sample.video_id in train_ids]
    val_idx = [i for i, sample in enumerate(all_samples) if sample.video_id in val_ids]
    test_idx = [i for i, sample in enumerate(all_samples) if sample.video_id in test_ids]

    print(
        f"[build_dataloaders] "
        f"학습: {len(train_ids)}개 비디오 / {len(train_idx)}샘플 | "
        f"검증: {len(val_ids)}개 비디오 / {len(val_idx)}샘플 | "
        f"테스트: {len(test_ids)}개 비디오 / {len(test_idx)}샘플"
    )
    _log_split_stats(all_samples, train_ids, val_ids, test_ids)

    train_ds = full_ds.subset(train_idx)
    val_ds = full_ds.subset(val_idx)
    test_ds = full_ds.subset(test_idx)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": TrafficAccidentQADataset.collate_fn,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
