"""범주형 메트릭 계산 함수"""
from typing import List, Dict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np


def compute_f1_score(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    F1, Precision, Recall 계산 (macro average)
    
    Args:
        y_true: 정답 레이블 리스트
        y_pred: 예측 레이블 리스트
        
    Returns:
        {
            "precision": float,
            "recall": float,
            "f1": float,
            "support": int  # 샘플 수
        }
    """
    if len(y_true) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": 0
        }
    
    # sklearn의 precision_recall_fscore_support 사용
    # average='macro': 각 클래스별 계산 후 평균 (클래스 불균형 고려)
    # zero_division=0: 경고 억제
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, 
        y_pred, 
        average='macro',
        zero_division=0
    )
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": int(np.sum(support)) if isinstance(support, np.ndarray) else len(y_true)
    }


def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """
    정확도 계산
    
    Args:
        y_true: 정답 레이블 리스트
        y_pred: 예측 레이블 리스트
        
    Returns:
        accuracy (0.0 ~ 1.0)
    """
    if len(y_true) == 0:
        return 0.0
    
    return float(accuracy_score(y_true, y_pred))


def compute_recall(y_true: List[str], y_pred: List[str]) -> float:
    """
    Recall 계산 (macro average)
    accident_place_feature 타입에서 강조 표시용
    
    Args:
        y_true: 정답 레이블 리스트
        y_pred: 예측 레이블 리스트
        
    Returns:
        recall (0.0 ~ 1.0)
    """
    if len(y_true) == 0:
        return 0.0
    
    _, recall, _, _ = precision_recall_fscore_support(
        y_true, 
        y_pred, 
        average='macro',
        zero_division=0
    )
    
    return float(recall)
