# scripts/train/train.py

import os
import json
import math
import shutil
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# =========================================================
# 기본 설정
# =========================================================
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"

TRAIN_JSON_PATH = "data/qa/train_flat.json"
VAL_JSON_PATH = "data/qa/val_flat.json"

OUTPUT_DIR = "outputs/qwen3_vl_4b_instruct_subset"
BEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "checkpoint-best")

SEED = 42

# Colab 기준 메모리 절약 설정
USE_4BIT = True
USE_LORA = True

# 영상 샘플링 설정
VIDEO_FPS = 1.0
VIDEO_NUM_FRAMES = None  # None이면 fps 기준 샘플링
MAX_NEW_TOKENS_LABEL = 64

# 학습 설정
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03

PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 3

SYSTEM_PROMPT = (
    "당신은 교통사고 영상 분석 전문가이다. "
    "주어진 영상과 질문을 보고 가장 적절한 답을 한국어로 간결하게 한 문장으로 답하라. "
    "답변은 불필요한 설명 없이 정답만 자연스럽게 작성하라. "
    "문체는 '~입니다.'로 통일하라."
)


# =========================================================
# 유틸
# =========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================================================
# 데이터셋
# =========================================================
class TrafficVideoQADataset(Dataset):
    def __init__(self, json_path: str):
        self.samples = load_json(json_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


# =========================================================
# Collator
# - 배치 크기 1 기준으로 설계
# - prompt 부분은 -100 마스킹
# - answer 부분만 loss 계산
# =========================================================
@dataclass
class Qwen3VLVideoQACollator:
    processor: Any
    system_prompt: str
    video_fps: float = 1.0
    video_num_frames: int = None
    max_label_tokens: int = 64

    def _build_prompt_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        video_content = {
            "type": "video",
            "url": sample["video_path"],
        }

        if self.video_num_frames is not None:
            video_content["num_frames"] = self.video_num_frames
        else:
            video_content["fps"] = self.video_fps

        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": sample["question"]},
                ],
            },
        ]

    def _build_full_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        answer_text = str(sample["answer"]).strip()

        # 너무 긴 정답 방지
        answer_text = answer_text[:512]

        prompt_messages = self._build_prompt_messages(sample)
        full_messages = prompt_messages + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text}],
            }
        ]
        return full_messages

    def _tokenize_prompt(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompt_messages = self._build_prompt_messages(sample)

        inputs = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        return inputs

    def _tokenize_full(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        full_messages = self._build_full_messages(sample)

        inputs = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        return inputs

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if len(features) != 1:
            raise ValueError(
                f"현재 collator는 batch_size=1만 지원합니다. 받은 batch size: {len(features)}"
            )

        sample = features[0]

        prompt_inputs = self._tokenize_prompt(sample)
        full_inputs = self._tokenize_full(sample)

        input_ids = full_inputs["input_ids"].clone()
        attention_mask = full_inputs["attention_mask"].clone()

        labels = input_ids.clone()

        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:, :prompt_len] = -100

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        batch = {}
        for k, v in full_inputs.items():
            batch[k] = v

        batch["labels"] = labels

        return batch


# =========================================================
# 모델 로드
# =========================================================
def load_model_and_processor():
    dtype = get_dtype()

    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    quant_config = None
    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype if not USE_4BIT else None,
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation="sdpa",
    )

    model.config.use_cache = False

    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)

    if USE_LORA:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model, processor


# =========================================================
# best 체크포인트 강제 저장
# =========================================================
def save_best_checkpoint(trainer: Trainer, processor: Any) -> None:
    ensure_dir(BEST_OUTPUT_DIR)

    # trainer는 load_best_model_at_end=True 덕분에 best weight를 메모리에 올린 상태
    trainer.model.save_pretrained(BEST_OUTPUT_DIR)
    processor.save_pretrained(BEST_OUTPUT_DIR)

    trainer_state_path = os.path.join(OUTPUT_DIR, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        shutil.copy2(
            trainer_state_path,
            os.path.join(BEST_OUTPUT_DIR, "trainer_state.json"),
        )

    best_info = {
        "base_model_name": MODEL_NAME,
        "best_model_checkpoint_from_trainer": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "best_output_dir": BEST_OUTPUT_DIR,
        "train_json_path": TRAIN_JSON_PATH,
        "val_json_path": VAL_JSON_PATH,
        "use_4bit": USE_4BIT,
        "use_lora": USE_LORA,
        "video_fps": VIDEO_FPS,
        "video_num_frames": VIDEO_NUM_FRAMES,
        "system_prompt": SYSTEM_PROMPT,
    }
    save_json(best_info, os.path.join(BEST_OUTPUT_DIR, "best_checkpoint_info.json"))


# =========================================================
# main
# =========================================================
def main():
    set_seed(SEED)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    ensure_dir(OUTPUT_DIR)

    if not os.path.exists(TRAIN_JSON_PATH):
        raise FileNotFoundError(f"train 파일이 없습니다: {TRAIN_JSON_PATH}")
    if not os.path.exists(VAL_JSON_PATH):
        raise FileNotFoundError(f"val 파일이 없습니다: {VAL_JSON_PATH}")

    train_dataset = TrafficVideoQADataset(TRAIN_JSON_PATH)
    val_dataset = TrafficVideoQADataset(VAL_JSON_PATH)

    print(f"[INFO] train size: {len(train_dataset)}")
    print(f"[INFO] val size  : {len(val_dataset)}")

    model, processor = load_model_and_processor()

    collator = Qwen3VLVideoQACollator(
        processor=processor,
        system_prompt=SYSTEM_PROMPT,
        video_fps=VIDEO_FPS,
        video_num_frames=VIDEO_NUM_FRAMES,
        max_label_tokens=MAX_NEW_TOKENS_LABEL,
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=SAVE_TOTAL_LIMIT,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        report_to="none",
        label_names=["labels"],
        optim="paged_adamw_8bit" if USE_4BIT else "adamw_torch",
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
    )

    train_result = trainer.train()

    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)

    eval_metrics = trainer.evaluate()
    trainer.save_metrics("eval", eval_metrics)

    save_best_checkpoint(trainer, processor)

    print("\n[INFO] 학습 완료")
    print(f"[INFO] best checkpoint (최종 저장 경로): {BEST_OUTPUT_DIR}")
    print(f"[INFO] trainer가 기록한 best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"[INFO] best metric: {trainer.state.best_metric}")
