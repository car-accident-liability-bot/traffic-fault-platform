import json
from pathlib import Path
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from traffic_ai_core.Qwen3_VL_4B_Instruct.model.model import load_model


TRAIN_PATH = Path("data/qa/processed/train_fault.json")
VAL_PATH = Path("data/qa/processed/val_fault.json")
OUTPUT_DIR = Path("checkpoints/qwen3vl_text_qa_sanity")


SYSTEM_PROMPT = (
    "너는 교통사고 QA 보조 모델이다. "
    "질문에 대해 가장 짧고 정확한 정답만 출력하라. "
    "불필요한 설명은 하지 마라."
)


def load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_text_prompt(question: str) -> str:
    return (
        f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n"
        f"[USER]\n{question}\n\n"
        f"[ASSISTANT]\n"
    )


class TrafficTextQADataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], processor, max_length: int = 256):
        self.data = data
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        question = sample["question"]
        answer = sample["answer"]

        prompt = build_text_prompt(question)
        full_text = prompt + answer

        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        labels = input_ids.clone()

        # prompt 부분은 loss 계산에서 제외하고, answer 부분만 학습하도록 처리
        prompt_encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_len = int(prompt_encoded["attention_mask"].sum().item())

        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def print_sample_examples(data: List[Dict[str, Any]], title: str, n: int = 3):
    print(f"\n[{title}] 샘플 예시")
    for i, sample in enumerate(data[:n]):
        print(f"- sample {i+1}")
        print(f"  question_type: {sample.get('question_type')}")
        print(f"  question     : {sample['question']}")
        print(f"  answer       : {sample['answer']}")


def main():
    print("[INFO] 데이터 로드 시작")
    train_data = load_json(TRAIN_PATH)
    val_data = load_json(VAL_PATH)

    print(f"[INFO] train size: {len(train_data)}")
    print(f"[INFO] val size  : {len(val_data)}")

    print_sample_examples(train_data, "train")
    print_sample_examples(val_data, "val")

    print("\n[INFO] 모델 로드 시작")
    model, processor = load_model()

    # tokenizer pad token 안전 처리
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # 학습 안정화
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    train_dataset = TrafficTextQADataset(train_data, processor)
    val_dataset = TrafficTextQADataset(val_data, processor)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
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
    )

    print("\n[INFO] 학습 시작")
    trainer.train()

    print("\n[INFO] 모델 저장")
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    print(f"[INFO] 저장 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()