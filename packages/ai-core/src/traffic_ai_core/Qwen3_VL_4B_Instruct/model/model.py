import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


class TrafficAccidentVLM:
    DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id  = model_id
        self.model     = None
        self.processor = None
        self._load()

    def _load(self):
        print(f"[TrafficAccidentVLM] 로드 중: {self.model_id}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype = torch.bfloat16,
            device_map  = "auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        print(f"[TrafficAccidentVLM] 완료 | "
              f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    def get_model(self):
        return self.model

    def get_processor(self):
        return self.processor


def load_model(model_id: str = TrafficAccidentVLM.DEFAULT_MODEL_ID):
    """
    편의 함수
    from ai_core import load_model
    model, processor = load_model()
    """
    vlm = TrafficAccidentVLM(model_id)
    return vlm.get_model(), vlm.get_processor()