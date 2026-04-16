from __future__ import annotations

import importlib.util

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class TrafficAccidentVLM:
    DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        prefer_flash_attention: bool = True,
    ):
        self.model_id = model_id
        self.prefer_flash_attention = prefer_flash_attention
        self.model = None
        self.processor = None
        self._load()

    @staticmethod
    def _pick_dtype() -> torch.dtype:
        # [수정] CUDA에서는 bf16 우선, 미지원 GPU면 fp16으로 내려 CPU RAM/VRAM 부담을 줄입니다.
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    def _pick_attn_implementation(self) -> str | None:
        # [수정] flash-attn이 있으면 우선 사용, 아니면 PyTorch SDPA로 fallback합니다.
        if not torch.cuda.is_available():
            return None
        if self.prefer_flash_attention and importlib.util.find_spec("flash_attn") is not None:
            return "flash_attention_2"
        return "sdpa"

    def _load(self):
        dtype = self._pick_dtype()
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        attn_implementation = self._pick_attn_implementation()

        print(f"[TrafficAccidentVLM] 로드 중: {self.model_id}")
        print(
            f"[TrafficAccidentVLM] dtype={dtype} | device={device_name} | "
            f"attn={attn_implementation or 'default'}"
        )

        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            # [수정] low_cpu_mem_usage=True로 Colab 환경에서 CPU 메모리 사용량을 줄입니다.
            "low_cpu_mem_usage": True,
        }
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **load_kwargs,
            )
        except TypeError:
            load_kwargs.pop("attn_implementation", None)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **load_kwargs,
            )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        # [수정] pad token / left padding을 명시해 collate 단계와 generate 단계 모두 안정화합니다.
        tokenizer = self.processor.tokenizer
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        print(f"[TrafficAccidentVLM] 완료 | device: {device_name}")

    def get_model(self):
        return self.model

    def get_processor(self):
        return self.processor


def load_model(
    model_id: str = TrafficAccidentVLM.DEFAULT_MODEL_ID,
    *,
    prefer_flash_attention: bool = True,
):
    vlm = TrafficAccidentVLM(
        model_id,
        prefer_flash_attention=prefer_flash_attention,
    )
    return vlm.get_model(), vlm.get_processor()
