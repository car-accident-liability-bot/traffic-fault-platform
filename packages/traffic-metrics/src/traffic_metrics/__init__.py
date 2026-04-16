"""traffic-metrics: 교통사고 QA 모델 평가 메트릭"""

from .evaluator import QAEvaluator
from .categorical import compute_f1_score, compute_accuracy, compute_recall
from .numerical import compute_fault_ratio_mae
from .utils import parse_fault_ratio, group_by_question_type

__version__ = "0.1.0"

__all__ = [
    # Main API
    "QAEvaluator",
    
    # Categorical metrics
    "compute_f1_score",
    "compute_accuracy",
    "compute_recall",
    
    # Numerical metrics
    "compute_fault_ratio_mae",
    
    # Utils
    "parse_fault_ratio",
    "group_by_question_type",
]
