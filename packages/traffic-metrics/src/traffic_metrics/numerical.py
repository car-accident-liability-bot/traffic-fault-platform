"""수치형 메트릭 계산 함수"""
from typing import List
import numpy as np
from .utils import parse_fault_ratio


def compute_fault_ratio_mae(predictions: List[str], ground_truths: List[str]) -> float:
    """
    과실 비율에 대한 MAE 계산 (차량 A 값만 사용)
    
    Args:
        predictions: 예측 과실 비율 리스트 (예: ["30:70", "50:50"])
        ground_truths: 정답 과실 비율 리스트 (예: ["30:70", "40:60"])
        
    Returns:
        MAE (Mean Absolute Error)
        
    Examples:
        >>> compute_fault_ratio_mae(["30:70", "50:50"], ["30:70", "40:60"])
        5.0
    """
    if len(predictions) == 0:
        return 0.0
    
    errors = []
    
    for pred, truth in zip(predictions, ground_truths):
        pred_a, _ = parse_fault_ratio(pred)
        truth_a, _ = parse_fault_ratio(truth)
        
        # 파싱 실패한 경우 (0, 0) 반환됨 -> 큰 오차로 기록
        if pred_a == 0 and truth_a == 0:
            # 둘 다 파싱 실패면 오차 0
            error = 0
        elif pred_a == 0 or truth_a == 0:
            # 하나만 파싱 실패면 최대 오차 (100)
            error = 100
        else:
            error = abs(pred_a - truth_a)
        
        errors.append(error)
    
    return float(np.mean(errors))
