import json
import random
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,cd /d D:\traffic-fault-platform
    TrainingArguments,
)

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
QA_DATA_PATH = "data/qa/qa_dataset.json"
OUTPUT_DIR = "checkpoints/qwen3vl_text_qa_sanity"

SYSTEM_PROMPT = (
    "너는 교통사고 QA 보조 모델이다. "
    "질문에 대해 가장 짧고 정확한 정답만 출력해라. "
    "불필요한 설명은 하지 마라."
)

SEED = 42
VAL_RATIO = 0.2
MAX_LENGTH = 256


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_by_video_id(data: List[Dict], val_ratio: float = 0.2, seed: int = 42):
    video_items = data[:]
    rng = random.Random(seed)
    rng.shuffle(video_items)

    n_total = len(video_items)
    n_val = max(1, int(n_total * val_ratio)) if n_total > 1 else 0

    val_items = video_items[:n_val]
    train_items = video_items[n_val:]

    if len(train_items) == 0 and len(val_items) > 0:
        train_items = val_items[:1]
        val_items = val_items[1:]

    return train_items, val_items


def flatten_qa(items: List[Dict]) -> List[Dict]:
    flat_samples = []

    for item in items:
        video_id = item["video_id"]
        qa_pairs = item.get("qa_pairs", [])

        for qa in qa_pairs:
            flat_samples.append(
                {
                    "video_id": video_id,
                    "question_type": qa["question_type"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                }
            )

    return flat_samples


def build_text_prompt(question: str) -> str:
    return (
        f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n"
        f"[USER]\n{question}\n\n"
        f"[ASSISTANT]\n"
    )


class TextQADataset(Dataset):
    def __init__(self, samples: List[Dict], processor, max_length: int = 256):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prompt = build_text_prompt(sample["question"])
        full_text = prompt + sample["answer"]

        enc = self.processor.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        prompt_enc = self.processor.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        prompt_len = len(prompt_enc["input_ids"])
        labels = input_ids[:]

        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class DataCollatorForCausalLM:
    pad_token_id: int

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len

            input_ids = f["input_ids"] + [self.pad_token_id] * pad_len
            attention_mask = f["attention_mask"] + [0] * pad_len
            labels = f["labels"] + [-100] * pad_len

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            labels_batch.append(labels)

        batch = {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }
        return batch


def main():
    set_seed(SEED)

    print("[INFO] 원본 QA 데이터 로드 시작")
    raw_data = load_json(QA_DATA_PATH)
    print(f"[INFO] video 샘플 수: {len(raw_data)}")

    train_items, val_items = split_by_video_id(raw_data, val_ratio=VAL_RATIO, seed=SEED)
    train_samples = flatten_qa(train_items)
    val_samples = flatten_qa(val_items)

    print(f"[INFO] train video 수: {len(train_items)}")
    print(f"[INFO] val video 수  : {len(val_items)}")
    print(f"[INFO] train qa 수   : {len(train_samples)}")
    print(f"[INFO] val qa 수     : {len(val_samples)}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = processor.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TextQADataset(train_samples, processor, max_length=MAX_LENGTH)
    val_dataset = TextQADataset(val_samples, processor, max_length=MAX_LENGTH) if len(val_samples) > 0 else None

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    model.config.use_cache = False

    data_collator = DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=5,
        learning_rate=2e-5,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset is not None else "no",
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("[INFO] 학습 시작")
    trainer.train()

    print("[INFO] 모델 저장")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print("[INFO] 완료")


if __name__ == "__main__":
    main()