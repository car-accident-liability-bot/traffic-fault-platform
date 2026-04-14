
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CACHE_PATH = Path("data/cache/label_inspection_cache.json")
DEFAULT_SUMMARY_PATH = Path("data/manifests/subset_pipeline_summary.json")
DEFAULT_MAPPING_XLSX_PATH = Path("data/reference/traffic_accident_fault_tables_v2.xlsx")
DEFAULT_OUTPUT_DIR = Path("data/eda")
DEFAULT_OUTPUT_FILE_NAME = "team_eda_report.md"
DEFAULT_RARE_THRESHOLD = 5
DEFAULT_TOP_N = 10


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _percent(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return (part / whole) * 100.0


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text if text else None


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 루트가 dict가 아닙니다: {path}")
    return payload


def _load_label_inspection_entries(cache_path: Path) -> list[dict[str, Any]]:
    payload = _load_json_file(cache_path)
    entries = payload.get("entries")

    if not isinstance(entries, dict):
        raise ValueError(f"'entries' 구조가 올바르지 않습니다: {cache_path}")

    normalized: list[dict[str, Any]] = []
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue

        normalized.append(
            {
                "has_valid_fault_ratio": bool(entry.get("has_valid_fault_ratio", False)),
                "road_type": _normalize_optional_str(entry.get("road_type")),
                "case_code": _normalize_optional_str(entry.get("case_code")),
                "has_minimum_required_fields": bool(entry.get("has_minimum_required_fields", False)),
            }
        )

    return normalized


def _find_first_dict_by_key(node: Any, target_key: str) -> dict[str, Any] | None:
    if isinstance(node, dict):
        if target_key in node and isinstance(node[target_key], dict):
            return node[target_key]
        for value in node.values():
            found = _find_first_dict_by_key(value, target_key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_first_dict_by_key(item, target_key)
            if found is not None:
                return found
    return None


def _extract_excluded_by_reason(summary_path: Path) -> Counter[str]:
    if not summary_path.exists():
        return Counter()

    summary = _load_json_file(summary_path)
    found = _find_first_dict_by_key(summary, "excluded_by_reason")
    if found is None:
        return Counter()

    result: Counter[str] = Counter()
    for key, value in found.items():
        try:
            result[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return result


def _detect_header_row(df: pd.DataFrame, required_tokens: set[str]) -> int:
    for row_index in range(min(len(df), 10)):
        row_values = {
            str(value).strip()
            for value in df.iloc[row_index].tolist()
            if not pd.isna(value) and str(value).strip()
        }
        if required_tokens.issubset(row_values):
            return row_index
    raise ValueError(f"헤더 행을 찾을 수 없습니다. required_tokens={sorted(required_tokens)}")


def _read_sheet_with_detected_header(
    xlsx_path: Path,
    sheet_name: str,
    required_tokens: set[str],
) -> pd.DataFrame:
    raw_df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    header_row_index = _detect_header_row(raw_df, required_tokens)
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=header_row_index)
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [str(column).strip() for column in df.columns]
    return df


def _normalize_code_key(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return str(int(value))
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def _build_simple_code_map(
    xlsx_path: Path,
    sheet_name: str,
    *,
    code_column: str = "클래스 코드",
    name_column: str = "구분",
) -> dict[str, str]:
    df = _read_sheet_with_detected_header(
        xlsx_path,
        sheet_name,
        required_tokens={code_column, name_column},
    )
    mapping: dict[str, str] = {}

    for _, row in df.iterrows():
        code_key = _normalize_code_key(row.get(code_column))
        name = _normalize_optional_str(row.get(name_column))
        if code_key is None or name is None:
            continue
        mapping[code_key] = name

    return mapping


def _build_case_code_description_map(xlsx_path: Path) -> dict[str, str]:
    required_tokens = {
        "사고객체",
        "사고장소",
        "사고장소특징",
        "A진행방향",
        "B진행방향",
        "과실비율A",
        "과실비율B",
        "사고유형",
    }
    df = _read_sheet_with_detected_header(
        xlsx_path,
        "사고유형434",
        required_tokens=required_tokens,
    )

    mapping: dict[str, str] = {}

    for _, row in df.iterrows():
        code_key = _normalize_code_key(row.get("사고유형"))
        if code_key is None:
            continue

        parts: list[str] = []
        accident_object = _normalize_optional_str(row.get("사고객체"))
        accident_place = _normalize_optional_str(row.get("사고장소"))
        accident_feature = _normalize_optional_str(row.get("사고장소특징"))
        vehicle_a = _normalize_optional_str(row.get("A진행방향"))
        vehicle_b = _normalize_optional_str(row.get("B진행방향"))

        ratio_a_raw = row.get("과실비율A")
        ratio_b_raw = row.get("과실비율B")
        ratio_a = _normalize_code_key(ratio_a_raw)
        ratio_b = _normalize_code_key(ratio_b_raw)

        if accident_object:
            parts.append(accident_object)
        if accident_place:
            parts.append(f"사고장소: {accident_place}")
        if accident_feature:
            parts.append(f"사고장소특징: {accident_feature}")
        if vehicle_a:
            parts.append(f"A진행방향: {vehicle_a}")
        if vehicle_b:
            parts.append(f"B진행방향: {vehicle_b}")
        if ratio_a is not None and ratio_b is not None:
            parts.append(f"과실비율: {ratio_a}:{ratio_b}")

        if parts:
            mapping[code_key] = " / ".join(parts)

    return mapping


def _build_mapping_bundle(xlsx_path: Path) -> dict[str, dict[str, str]]:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"엑셀 매핑 파일이 없습니다: {xlsx_path}")

    place_map = _build_simple_code_map(xlsx_path, "모델1_place")
    feature_map = _build_simple_code_map(xlsx_path, "모델2_feature")
    vehicle_a_map = _build_simple_code_map(xlsx_path, "모델3_vehicleA")
    vehicle_b_map = _build_simple_code_map(xlsx_path, "모델4_vehicleB")
    case_code_map = _build_case_code_description_map(xlsx_path)

    return {
        "place_map": place_map,
        "feature_map": feature_map,
        "vehicle_a_map": vehicle_a_map,
        "vehicle_b_map": vehicle_b_map,
        "case_code_map": case_code_map,
    }


def _extract_road_codes(road_type_value: str) -> tuple[str | None, str | None]:
    place_match = re.search(r"accident_place:(\d+)", road_type_value)
    feature_match = re.search(r"accident_place_feature:(\d+)", road_type_value)
    place_code = place_match.group(1) if place_match else None
    feature_code = feature_match.group(1) if feature_match else None
    return place_code, feature_code


def _format_case_code_label(case_code: str, case_code_map: dict[str, str]) -> str:
    description = case_code_map.get(case_code)
    if description:
        return f"{case_code} ({description})"
    return case_code


def _escape_markdown_table_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _format_road_type_label(
    road_type_value: str,
    place_map: dict[str, str],
    feature_map: dict[str, str],
) -> str:
    place_code, feature_code = _extract_road_codes(road_type_value)

    place_name = place_map.get(place_code) if place_code is not None else None
    feature_name = feature_map.get(feature_code) if feature_code is not None else None

    details: list[str] = []
    if place_code is not None:
        if place_name:
            details.append(f"accident_place:{place_code} ({place_name})")
        else:
            details.append(f"accident_place:{place_code}")
    if feature_code is not None:
        if feature_name:
            details.append(f"accident_place_feature:{feature_code} ({feature_name})")
        else:
            details.append(f"accident_place_feature:{feature_code}")

    if details:
        return " / ".join(details)

    return road_type_value


def _build_distribution_lines(
    counter: Counter[str],
    total_count: int,
    *,
    top_n: int,
    label_formatter,
    key_name: str,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"| 순위 | {key_name} | count | ratio_percent | cumulative_ratio_percent |")
    lines.append("|---:|---|---:|---:|---:|")

    cumulative = 0
    for rank, (key, count) in enumerate(counter.most_common(top_n), start=1):
        cumulative += count
        lines.append(
            f"| {rank} | {_escape_markdown_table_cell(label_formatter(key))} | {count:,} | "
            f"{_percent(count, total_count):.2f} | {_percent(cumulative, total_count):.2f} |"
        )
    return lines


def _build_rare_case_lines(
    counter: Counter[str],
    total_count: int,
    *,
    label_formatter,
) -> list[str]:
    lines: list[str] = []
    if not counter:
        lines.append("- 희귀 case_code 없음")
        return lines

    lines.append("| 순위 | case_code | count | ratio_percent |")
    lines.append("|---:|---|---:|---:|")
    for rank, (code, count) in enumerate(counter.most_common(), start=1):
        lines.append(
            f"| {rank} | {_escape_markdown_table_cell(label_formatter(code))} | {count:,} | {_percent(count, total_count):.2f} |"
        )
    return lines


def _build_excluded_reason_lines(counter: Counter[str], total_count: int) -> list[str]:
    lines: list[str] = []
    if not counter:
        lines.append("- 제외 사유 데이터 없음")
        return lines

    lines.append("| reason | count | ratio_percent |")
    lines.append("|---|---:|---:|")
    for reason, count in counter.most_common():
        lines.append(
            f"| {reason} | {count:,} | {_percent(count, total_count):.2f} |"
        )
    return lines


def _build_highlight_lines(
    case_code_counter: Counter[str],
    road_type_counter: Counter[str],
    total_count: int,
    *,
    case_label_formatter,
    road_label_formatter,
) -> list[str]:
    lines: list[str] = []

    top_case = case_code_counter.most_common(1)
    if top_case:
        code, count = top_case[0]
        lines.append(
            f"- 가장 많은 case_code는 **{case_label_formatter(code)}** 이고, "
            f"전체의 **{_percent(count, total_count):.2f}%**를 차지합니다."
        )

    top_road = road_type_counter.most_common(1)
    if top_road:
        road_type, count = top_road[0]
        lines.append(
            f"- 가장 많은 road_type은 **{road_label_formatter(road_type)}** 이고, "
            f"전체의 **{_percent(count, total_count):.2f}%**를 차지합니다."
        )

    if len(case_code_counter) > 0:
        rare_total = sum(count for _code, count in case_code_counter.items() if count < 5)
        lines.append(
            f"- count < 5 기준 희귀 case_code 표본 수는 **{rare_total:,}개**입니다."
        )

    return lines


def _write_markdown_report(
    output_path: Path,
    *,
    total_inspected: int,
    case_code_counter: Counter[str],
    road_type_counter: Counter[str],
    rare_case_counter: Counter[str],
    excluded_by_reason: Counter[str],
    valid_fault_ratio_true: int,
    minimum_required_true: int,
    rare_threshold: int,
    top_n: int,
    mapping_bundle: dict[str, dict[str, str]],
    mapping_xlsx_path: Path,
) -> None:
    _ensure_dir(output_path.parent)

    case_code_map = mapping_bundle["case_code_map"]
    place_map = mapping_bundle["place_map"]
    feature_map = mapping_bundle["feature_map"]

    case_label_formatter = lambda key: _format_case_code_label(key, case_code_map)
    road_label_formatter = lambda key: _format_road_type_label(key, place_map, feature_map)

    lines: list[str] = []
    lines.append("# 팀 공유용 EDA 리포트")
    lines.append("")
    lines.append("## 1. 개요")
    lines.append("")
    lines.append(f"- 전체 검사 라벨 수: **{total_inspected:,}**")
    lines.append(f"- 고유 case_code 수: **{len(case_code_counter):,}**")
    lines.append(f"- 고유 road_type 수: **{len(road_type_counter):,}**")
    lines.append(f"- 희귀 case_code 기준: **count < {rare_threshold}**")
    lines.append(f"- 희귀 case_code 종류 수: **{len(rare_case_counter):,}**")
    lines.append(f"- 희귀 case_code 전체 표본 수: **{sum(rare_case_counter.values()):,}**")
    lines.append("")
    lines.append("## 2. 데이터 품질")
    lines.append("")
    lines.append(
        f"- has_valid_fault_ratio=True: **{valid_fault_ratio_true:,}** "
        f"({_percent(valid_fault_ratio_true, total_inspected):.2f}%)"
    )
    lines.append(
        f"- has_minimum_required_fields=True: **{minimum_required_true:,}** "
        f"({_percent(minimum_required_true, total_inspected):.2f}%)"
    )
    lines.append("")
    lines.append("## 3. 발표용 핵심 포인트")
    lines.append("")
    lines.extend(_build_highlight_lines(
        case_code_counter,
        road_type_counter,
        total_inspected,
        case_label_formatter=case_label_formatter,
        road_label_formatter=road_label_formatter,
    ) or ["- 핵심 포인트를 만들 데이터가 없습니다."])
    lines.append("")
    lines.append(f"## 4. case_code 상위 {top_n}개")
    lines.append("")
    lines.extend(
        _build_distribution_lines(
            case_code_counter,
            total_inspected,
            top_n=top_n,
            label_formatter=case_label_formatter,
            key_name="case_code",
        )
        if case_code_counter
        else ["- case_code 데이터 없음"]
    )
    lines.append("")
    lines.append(f"## 5. road_type 상위 {top_n}개")
    lines.append("")
    lines.extend(
        _build_distribution_lines(
            road_type_counter,
            total_inspected,
            top_n=top_n,
            label_formatter=road_label_formatter,
            key_name="road_type",
        )
        if road_type_counter
        else ["- road_type 데이터 없음"]
    )
    lines.append("")
    lines.append("## 6. 희귀 case_code 목록")
    lines.append("")
    lines.extend(
        _build_rare_case_lines(
            rare_case_counter,
            total_inspected,
            label_formatter=case_label_formatter,
        )
    )
    lines.append("")
    lines.append("## 7. 제외 사유")
    lines.append("")
    lines.extend(_build_excluded_reason_lines(excluded_by_reason, total_inspected))
    lines.append("")
    lines.append("## 8. 코드 해석 기준")
    lines.append("")
    lines.append(f"- case_code 한글 설명 출처: `{mapping_xlsx_path}` 의 `사고유형434` 시트")
    lines.append(f"- road_type 한글 설명 출처: `{mapping_xlsx_path}` 의 `모델1_place`, `모델2_feature` 시트")
    lines.append("- Markdown에는 **원본 코드값을 유지**하면서, 괄호 안에 한글 설명을 병기했습니다.")
    lines.append("- case_code는 단순 이름 대신 **사고장소/특징/A진행방향/B진행방향/과실비율**을 묶은 설명문으로 표시했습니다.")
    lines.append("- road_type는 `accident_place:x|accident_place_feature:y` 형식을 유지하면서, 각 코드 옆에 한글명을 붙였습니다.")
    lines.append("")

    with output_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="코드표 엑셀을 반영해 팀 공유용 Markdown EDA 리포트를 생성합니다."
    )
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--mapping-xlsx-path", type=Path, default=DEFAULT_MAPPING_XLSX_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-file-name", type=str, default=DEFAULT_OUTPUT_FILE_NAME)
    parser.add_argument("--rare-threshold", type=int, default=DEFAULT_RARE_THRESHOLD)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args()

    inspections = _load_label_inspection_entries(args.cache_path)
    excluded_by_reason = _extract_excluded_by_reason(args.summary_path)
    mapping_bundle = _build_mapping_bundle(args.mapping_xlsx_path)

    total_inspected = len(inspections)
    case_code_counter: Counter[str] = Counter(
        item["case_code"] for item in inspections if item["case_code"] is not None
    )
    road_type_counter: Counter[str] = Counter(
        item["road_type"] for item in inspections if item["road_type"] is not None
    )
    rare_case_counter: Counter[str] = Counter(
        {
            code: count
            for code, count in case_code_counter.items()
            if count < args.rare_threshold
        }
    )

    valid_fault_ratio_true = sum(1 for item in inspections if item["has_valid_fault_ratio"])
    minimum_required_true = sum(1 for item in inspections if item["has_minimum_required_fields"])

    output_path = args.output_dir / args.output_file_name
    _write_markdown_report(
        output_path,
        total_inspected=total_inspected,
        case_code_counter=case_code_counter,
        road_type_counter=road_type_counter,
        rare_case_counter=rare_case_counter,
        excluded_by_reason=excluded_by_reason,
        valid_fault_ratio_true=valid_fault_ratio_true,
        minimum_required_true=minimum_required_true,
        rare_threshold=args.rare_threshold,
        top_n=args.top_n,
        mapping_bundle=mapping_bundle,
        mapping_xlsx_path=args.mapping_xlsx_path,
    )

    print("[EDA] 완료")
    print(f"[EDA] output_path={output_path}")
    print(f"[EDA] total_inspected_items={total_inspected}")
    print(f"[EDA] case_code_unique_count={len(case_code_counter)}")
    print(f"[EDA] road_type_unique_count={len(road_type_counter)}")


if __name__ == "__main__":
    main()
