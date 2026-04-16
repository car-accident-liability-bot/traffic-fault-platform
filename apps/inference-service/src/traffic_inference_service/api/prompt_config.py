SYSTEM_PROMPT = (
    "당신은 교통사고 영상을 보고 질문에 대해 짧고 정확하게 답하는 AI입니다. "
    "반드시 질문에 대한 핵심 답만 간단히 출력하세요. "
    "불필요한 설명은 하지 마세요."
)

QUESTION_MAP = {
    "accident_place": "이 사고는 어떤 도로 환경에서 발생했는가?",
    "accident_place_feature": "이 사고 장소의 특징은 무엇인가?",
    "vehicle_a_progress": "차량 A는 사고 직전 어떤 진행 상태였는가?",
    "vehicle_b_progress": "차량 B는 사고 직전 어떤 진행 상태였는가?",
    "fault_ratio": "이 사고의 과실비율은 어떻게 되는가?",
    "fault_compare": "과실비율 기준으로 더 큰 과실을 가진 차량은 누구인가?",
}
