"""QA 평가 메인 클래스"""
from typing import List, Dict, Any
from .utils import group_by_question_type
from .categorical import compute_f1_score, compute_accuracy, compute_recall
from .numerical import compute_fault_ratio_mae


class QAEvaluator:
    """
    교통사고 QA 모델 평가기
    
    입력 형식:
        [{
            "video_id": str,
            "question_type": str,
            "prediction": str,
            "ground_truth": str
        }, ...]
    
    출력 형식:
        {
            "overall": {
                "macro_f1": float,
                "total_samples": int,
                "accuracy": float
            },
            "per_type": {
                "accident_place": {
                    "f1": float,
                    "precision": float,
                    "recall": float,
                    "support": int
                },
                ...
            }
        }
    """
    
    def __init__(self):
        """QAEvaluator 초기화"""
        pass
    
    def evaluate(self, data: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        평가 실행
        
        Args:
            data: 평가 데이터 리스트
                [{video_id, question_type, prediction, ground_truth}, ...]
        
        Returns:
            평가 결과 딕셔너리
        """
        if not data or len(data) == 0:
            return {
                "overall": {
                    "macro_f1": 0.0,
                    "total_samples": 0,
                    "accuracy": 0.0
                },
                "per_type": {}
            }
        
        # 질문 타입별로 그룹화
        grouped = group_by_question_type(data)
        
        # 각 타입별 평가
        per_type_results = {}
        f1_scores = []
        
        for q_type, samples in grouped.items():
            y_true = [s["ground_truth"] for s in samples]
            y_pred = [s["prediction"] for s in samples]
            
            # 기본 F1/Precision/Recall 계산
            metrics = compute_f1_score(y_true, y_pred)
            per_type_results[q_type] = metrics
            f1_scores.append(metrics["f1"])
            
            # 타입별 추가 메트릭
            if q_type == "accident_place_feature":
                # Recall 강조
                metrics["recall_highlighted"] = metrics["recall"]
            
            elif q_type == "fault_ratio":
                # MAE 추가
                mae = compute_fault_ratio_mae(y_pred, y_true)
                metrics["mae"] = mae
            
            elif q_type in ["accident_place", "fault_compare"]:
                # Accuracy 추가 (선택)
                acc = compute_accuracy(y_true, y_pred)
                metrics["accuracy"] = acc
        
        # 전체 메트릭 계산
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        # 전체 정확도 (모든 샘플)
        all_true = [s["ground_truth"] for s in data]
        all_pred = [s["prediction"] for s in data]
        overall_accuracy = compute_accuracy(all_true, all_pred)
        
        return {
            "overall": {
                "macro_f1": macro_f1,
                "total_samples": len(data),
                "accuracy": overall_accuracy
            },
            "per_type": per_type_results
        }
