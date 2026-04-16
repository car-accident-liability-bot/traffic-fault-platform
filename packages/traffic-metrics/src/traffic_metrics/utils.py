"""평가 유틸리티 함수"""
from typing import Dict, List
import re


def parse_fault_ratio(text: str) -> tuple[int, int]:
    """
    과실 비율 텍스트를 파싱하여 (A, B) 튜플 반환
    
    Args:
        text: "30:70" 형식의 문자열
        
    Returns:
        (A_ratio, B_ratio) 튜플. 예: (30, 70)
        
    Examples:
        >>> parse_fault_ratio("30:70")
        (30, 70)
        >>> parse_fault_ratio("100:0")
        (100, 0)
    """
    text = text.strip()
    
    # "30:70" 형식 매칭
    match = re.match(r'(\d+)\s*:\s*(\d+)', text)
    if match:
        a_ratio = int(match.group(1))
        b_ratio = int(match.group(2))
        return (a_ratio, b_ratio)
    
    # 파싱 실패 시 (0, 0) 반환 (에러 핸들링)
    return (0, 0)


def group_by_question_type(data: List[Dict]) -> Dict[str, List[Dict]]:
    """
    평가 데이터를 question_type별로 그룹화
    
    Args:
        data: [{video_id, question_type, prediction, ground_truth}, ...]
        
    Returns:
        {question_type: [samples], ...}
        
    Examples:
        >>> data = [
        ...     {"question_type": "accident_place", "prediction": "A", "ground_truth": "A"},
        ...     {"question_type": "accident_place", "prediction": "B", "ground_truth": "B"},
        ...     {"question_type": "fault_ratio", "prediction": "30:70", "ground_truth": "30:70"}
        ... ]
        >>> grouped = group_by_question_type(data)
        >>> len(grouped["accident_place"])
        2
        >>> len(grouped["fault_ratio"])
        1
    """
    grouped = {}
    
    for item in data:
        q_type = item["question_type"]
        if q_type not in grouped:
            grouped[q_type] = []
        grouped[q_type].append(item)
    
    return grouped
