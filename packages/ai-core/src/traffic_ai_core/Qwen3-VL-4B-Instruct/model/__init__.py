# ai_core/__init__.py
# 역할: 패키지 진입점
# 외부에서 `from ai_core import load_model, TrafficAccidentInference` 로 사용

from ai_core.model import load_model, TrafficAccidentVLM
from ai_core.inference import TrafficAccidentInference