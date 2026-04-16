import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


class TrafficAccidentVLM:
    DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._load()

    def _load(self):
        print(f"[TrafficAccidentVLM] 로드 중: {self.model_id}")

        if torch.cuda.is_available():
            dtype = torch.bfloat16
            device_name = "cuda"
        else:
            dtype = torch.float32
            device_name = "cpu"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        print(f"[TrafficAccidentVLM] 완료 | device: {device_name}")

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