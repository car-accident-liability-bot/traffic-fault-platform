import json
import os
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

LABEL_BASE_URL = "https://data.taeo-dev.com/dataset/traffic/subset/label/"
CODEBOOK_PATH = "./data/qa/traffic_accident_tables.xlsx"
OUTPUT_PATH = "./data/qa/qa_dataset_3k.json"

def load_codebook():
    xl = pd.ExcelFile(CODEBOOK_PATH)

    df_place = xl.parse("모델1_accident_place")
    place_map = dict(zip(df_place["클래스 코드"], df_place["구분"]))

    df_feature = xl.parse("모델2_accident_place_feature")
    feature_map = dict(zip(df_feature["클래스 코드"], df_feature["구분"]))

    df_a = xl.parse("모델3_vehicle_a_progress")
    a_map = dict(zip(df_a["클래스 코드"], df_a["구분"]))

    df_b = xl.parse("모델4_vehicle_b_progress")
    b_map = dict(zip(df_b["클래스 코드"], df_b["구분"]))

    return place_map, feature_map, a_map, b_map


place_map, feature_map, a_map, b_map = load_codebook()

QUESTION_TEMPLATES = {
    "accident_place": "이 사고는 어떤 장소에서 발생했는가?",
    "accident_place_feature": "이 사고 장소의 특징은 무엇인가?",
    "vehicle_a_progress": "차량 A의 주행 상태는 무엇이었는가?",
    "vehicle_b_progress": "차량 B의 주행 상태는 무엇이었는가?",
    "fault_ratio": "차량 A와 차량 B의 과실비율은 얼마인가?",
    "fault_compare": "차량 A와 차량 B 중 어느 쪽이 더 큰 과실을 가지는가?",
}

def get_links(url: str):
    res = requests.get(url, timeout=30)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")
    links = []

    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        if href in ("../", "./"):
            continue
        links.append(href)

    return links


def crawl_json_urls(start_url: str):
    visited = set()
    json_urls = []

    def _crawl(url: str):
        if url in visited:
            return
        visited.add(url)

        print(f"[CRAWL] {url}")

        try:
            links = get_links(url)
        except Exception as e:
            print(f"[ERROR] 링크 조회 실패: {url} -> {e}")
            return

        for href in links:
            full_url = urljoin(url, href)

            if href.endswith(".json"):
                json_urls.append(full_url)
            elif href.endswith("/"):
                _crawl(full_url)

    _crawl(start_url)
    return sorted(json_urls)


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


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def map_value(value, mapping):
    if is_missing(value):
        return None

    try:
        int_value = int(value)
        return mapping.get(int_value, str(int_value))
    except Exception:
        return str(value)


def parse_fault_value(value):
    if is_missing(value):
        return None

    try:
        return int(float(value))
    except Exception:
        return None


def make_qa_from_data(data, video_id):
    raw_place = find_value(data, ["accident_place", "road_type"])
    raw_feature = find_value(data, ["accident_place_feature"])
    raw_a = find_value(data, ["vehicle_a_progress_info"])
    raw_b = find_value(data, ["vehicle_b_progress_info"])

    raw_fault_a = find_value(data, ["accident_negligence_rateA"])
    raw_fault_b = find_value(data, ["accident_negligence_rateB"])

    accident_place = map_value(raw_place, place_map)
    accident_feature = map_value(raw_feature, feature_map)
    vehicle_a = map_value(raw_a, a_map)
    vehicle_b = map_value(raw_b, b_map)

    fault_a = parse_fault_value(raw_fault_a)
    fault_b = parse_fault_value(raw_fault_b)

    qa_pairs = []

    if accident_place is not None:
        qa_pairs.append({
            "question_type": "accident_place",
            "question": QUESTION_TEMPLATES["accident_place"],
            "answer": accident_place,
        })

    if accident_feature is not None:
        qa_pairs.append({
            "question_type": "accident_place_feature",
            "question": QUESTION_TEMPLATES["accident_place_feature"],
            "answer": accident_feature,
        })

    if vehicle_a is not None:
        qa_pairs.append({
            "question_type": "vehicle_a_progress",
            "question": QUESTION_TEMPLATES["vehicle_a_progress"],
            "answer": vehicle_a,
        })

    if vehicle_b is not None:
        qa_pairs.append({
            "question_type": "vehicle_b_progress",
            "question": QUESTION_TEMPLATES["vehicle_b_progress"],
            "answer": vehicle_b,
        })

    if fault_a is not None and fault_b is not None:
        qa_pairs.append({
            "question_type": "fault_ratio",
            "question": QUESTION_TEMPLATES["fault_ratio"],
            "answer": f"{fault_a}:{fault_b}",
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
            "answer": comp,
        })

    if not qa_pairs:
        return None

    return {
        "video_id": video_id,
        "qa_pairs": qa_pairs,
    }


def main():
    if not os.path.isfile(CODEBOOK_PATH):
        raise FileNotFoundError(f"코드북 파일 없음: {CODEBOOK_PATH}")

    json_urls = crawl_json_urls(LABEL_BASE_URL)
    print(f"\n[INFO] 찾은 json URL 수: {len(json_urls)}")

    qa_dataset = []
    error_count = 0

    for i, json_url in enumerate(json_urls, start=1):
        try:
            res = requests.get(json_url, timeout=60)
            res.raise_for_status()
            data = res.json()

            video_id = json_url.rstrip("/").split("/")[-1]

            result = make_qa_from_data(data, video_id)
            if result:
                qa_dataset.append(result)

        except Exception as e:
            error_count += 1
            print(f"[ERROR] {json_url} -> {e}")

        if i % 100 == 0 or i == len(json_urls):
            print(f"[INFO] 진행: {i}/{len(json_urls)} | 현재 QA 샘플 수: {len(qa_dataset)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, ensure_ascii=False, indent=2)

    print("\n[완료]")
    print("생성된 QA 샘플 수:", len(qa_dataset))
    print("에러 수:", error_count)
    print("저장 위치:", OUTPUT_PATH)


if __name__ == "__main__":
    main()