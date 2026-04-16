# traffic-metrics

교통사고 QA 모델 평가를 위한 메트릭 패키지

## 설치

```bash
pip install -e .
```

## 사용법

```python
from traffic_metrics import QAEvaluator

# 평가 데이터 준비
evaluation_data = [
    {
        "video_id": "video_001.json",
        "question_type": "accident_place",
        "prediction": "T자형 교차로",
        "ground_truth": "T자형 교차로"
    },
    # ...
]

# 평가 실행
evaluator = QAEvaluator()
results = evaluator.evaluate(evaluation_data)

print(f"Macro F1: {results['overall']['macro_f1']:.4f}")
```

## 지원하는 메트릭

- **Macro F1**: 모든 질문 타입에 대한 평균 F1 Score
- **F1/Precision/Recall**: 각 질문 타입별
- **MAE**: fault_ratio 타입 전용
- **Recall**: accident_place_feature 타입 강조

## 질문 타입

- `accident_place`: 사고 장소
- `accident_place_feature`: 사고 장소 특징
- `vehicle_a_progress`: 차량 A 주행 상태
- `vehicle_b_progress`: 차량 B 주행 상태
- `fault_ratio`: 과실 비율
- `fault_compare`: 과실 비교
