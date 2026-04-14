import os
import json
import pandas as pd

# =========================================================
# 1. 경로 설정
# =========================================================
JSON_ROOT_DIR = "./data/working/pipeline_subset/label"
CODEBOOK_PATH = "./data/qa/traffic_accident_tables.xlsx"
OUTPUT_PATH = "./data/qa/qa_dataset.json"

# =========================================================
# 2. 코드북 로딩
# =========================================================
def load_codebook():
    xl = pd.ExcelFile(CODEBOOK_PATH)

    # 사고 장소
    df_place = xl.parse("모델1_accident_place")
    place_map = dict(zip(df_place["클래스 코드"], df_place["구분"]))

    # 사고 특징
    df_feature = xl.parse("모델2_accident_place_feature")
    feature_map = dict(zip(df_feature["클래스 코드"], df_feature["구분"]))

    # 차량 A 진행
    df_a = xl.parse("모델3_vehicle_a_progress")
    a_map = dict(zip(df_a["클래스 코드"], df_a["구분"]))

    # 차량 B 진행
    df_b = xl.parse("모델4_vehicle_b_progress")
    b_map = dict(zip(df_b["클래스 코드"], df_b["구분"]))

    return place_map, feature_map, a_map, b_map

place_map, feature_map, a_map, b_map = load_codebook()

# =========================================================
# 3. 질문 템플릿
# =========================================================
QUESTION_TEMPLATES = {
    "accident_place": "이 사고는 어떤 장소에서 발생했는가?",
    "accident_place_feature": "이 사고 장소의 특징은 무엇인가?",
    "vehicle_a_progress": "차량 A의 주행 상태는 무엇이었는가?",
    "vehicle_b_progress": "차량 B의 주행 상태는 무엇이었는가?",
    "fault_ratio": "차량 A와 차량 B의 과실비율은 얼마인가?",
    "fault_compare": "차량 A와 차량 B 중 어느 쪽이 더 큰 과실을 가지는가?"
}

# =========================================================
# 4. 유틸 함수
# =========================================================
def find_value(data, keys):
    if isinstance(data, dict):
        for k, v in data.items():
            if k in keys:
                return v
            result = find_value(v, keys)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_value(item, keys)
            if result is not None:
                return result
    return None

def map_value(value, mapping):
    try:
        return mapping.get(int(value), str(value))
    except:
        return str(value)

# =========================================================
# 5. QA 생성
# =========================================================
def make_qa_from_json(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    video_id = os.path.basename(json_path)

    # 원본 값
    raw_place = find_value(data, ["accident_place", "road_type"])
    raw_feature = find_value(data, ["accident_place_feature"])
    raw_a = find_value(data, ["vehicle_a_progress_info"])
    raw_b = find_value(data, ["vehicle_b_progress_info"])

    fault_a = find_value(data, ["accident_negligence_rateA"])
    fault_b = find_value(data, ["accident_negligence_rateB"])

    # 🔥 코드북 변환
    accident_place = map_value(raw_place, place_map)
    accident_feature = map_value(raw_feature, feature_map)
    vehicle_a = map_value(raw_a, a_map)
    vehicle_b = map_value(raw_b, b_map)

    qa_pairs = []

    if accident_place:
        qa_pairs.append({
            "question_type": "accident_place",
            "question": QUESTION_TEMPLATES["accident_place"],
            "answer": accident_place
        })

    if accident_feature:
        qa_pairs.append({
            "question_type": "accident_place_feature",
            "question": QUESTION_TEMPLATES["accident_place_feature"],
            "answer": accident_feature
        })

    if vehicle_a:
        qa_pairs.append({
            "question_type": "vehicle_a_progress",
            "question": QUESTION_TEMPLATES["vehicle_a_progress"],
            "answer": vehicle_a
        })

    if vehicle_b:
        qa_pairs.append({
            "question_type": "vehicle_b_progress",
            "question": QUESTION_TEMPLATES["vehicle_b_progress"],
            "answer": vehicle_b
        })

    if fault_a is not None and fault_b is not None:
        fault_a = int(fault_a)
        fault_b = int(fault_b)

        ratio = f"{fault_a}:{fault_b}"

        qa_pairs.append({
            "question_type": "fault_ratio",
            "question": QUESTION_TEMPLATES["fault_ratio"],
            "answer": ratio
        })

        if fault_a > fault_b:
            comp = "차량 A"
        elif fault_b > fault_a:
            comp = "차량 B"
        else:
            comp = "동일"

        qa_pairs.append({
            "question_type": "fault_compare",
            "question": QUESTION_TEMPLATES["fault_compare"],
            "answer": comp
        })

    if not qa_pairs:
        return None

    return {
        "video_id": video_id,
        "qa_pairs": qa_pairs
    }

# =========================================================
# 6. 전체 실행
# =========================================================
def main():
    qa_dataset = []

    for root, _, files in os.walk(JSON_ROOT_DIR):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)

                try:
                    result = make_qa_from_json(path)
                    if result:
                        qa_dataset.append(result)
                except Exception as e:
                    print(f"[ERROR] {file}: {e}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, ensure_ascii=False, indent=2)

    print("완료")
    print("총 개수:", len(qa_dataset))
    print("저장 위치:", OUTPUT_PATH)

if __name__ == "__main__":
    main()