from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_CACHE_PATH = Path("data/cache/label_inspection_cache.json")
DEFAULT_SUMMARY_PATH = Path("data/manifests/subset_pipeline_summary.json")
DEFAULT_OUTPUT_DIR = Path("data/eda")
DEFAULT_OUTPUT_FILE_NAME = "team_eda_report.md"
DEFAULT_RARE_THRESHOLD = 5
DEFAULT_TOP_N = 10


# 수정 설명:
# - 기존처럼 CSV/PNG/JSON 여러 파일을 만들지 않고, Markdown 1개만 생성합니다.
# - 팀원이 바로 읽기 쉽게 전체 개요, 품질 요약, 상위 분포, 희귀 코드, 제외 사유를 한 문서에 정리합니다.
# - 키값(case_code, road_type)은 그대로 사용합니다.


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 루트가 dict가 아닙니다: {path}")
    return payload


def normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def percent(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return (part / whole) * 100.0


def load_label_inspection_entries(cache_path: Path) -> list[dict[str, Any]]:
    payload = load_json_file(cache_path)
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"'entries' 구조가 올바르지 않습니다: {cache_path}")

    normalized: list[dict[str, Any]] = []
    for value in entries.values():
        if not isinstance(value, dict):
            continue

        normalized.append(
            {
                "has_valid_fault_ratio": bool(value.get("has_valid_fault_ratio", False)),
                "road_type": normalize_optional_str(value.get("road_type")),
                "case_code": normalize_optional_str(value.get("case_code")),
                "has_minimum_required_fields": bool(value.get("has_minimum_required_fields", False)),
            }
        )

    return normalized


def find_first_dict_by_key(node: Any, target_key: str) -> dict[str, Any] | None:
    if isinstance(node, dict):
        if target_key in node and isinstance(node[target_key], dict):
            return node[target_key]
        for value in node.values():
            found = find_first_dict_by_key(value, target_key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = find_first_dict_by_key(item, target_key)
            if found is not None:
                return found
    return None


def extract_excluded_by_reason(summary_path: Path) -> Counter[str]:
    if not summary_path.exists():
        return Counter()

    summary = load_json_file(summary_path)
    found = find_first_dict_by_key(summary, "excluded_by_reason")
    if found is None:
        return Counter()

    result = Counter()
    for key, value in found.items():
        try:
            result[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return result


def build_rank_rows(counter: Counter[str], total: int, top_n: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative = 0

    items = counter.most_common(top_n)
    for rank, (key, count) in enumerate(items, start=1):
        cumulative += count
        rows.append(
            {
                "rank": rank,
                "key": key,
                "count": count,
                "ratio_percent": percent(count, total),
                "cumulative_ratio_percent": percent(cumulative, total),
            }
        )
    return rows


def markdown_table(
    rows: list[dict[str, Any]],
    key_header: str,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"| rank | {key_header} | count | ratio_percent | cumulative_ratio_percent |")
    lines.append(f"|---:|---|---:|---:|---:|")

    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['key']} | {row['count']:,} | {row['ratio_percent']:.2f} | {row['cumulative_ratio_percent']:.2f} |"
        )

    if not rows:
        lines.append("| - | 데이터 없음 | - | - | - |")

    return lines


def markdown_reason_table(excluded_by_reason: Counter[str], total: int) -> list[str]:
    lines: list[str] = []
    lines.append("| rank | reason | count | ratio_percent |")
    lines.append("|---:|---|---:|---:|")

    if not excluded_by_reason:
        lines.append("| - | 데이터 없음 | - | - |")
        return lines

    for rank, (reason, count) in enumerate(excluded_by_reason.most_common(), start=1):
        lines.append(f"| {rank} | {reason} | {count:,} | {percent(count, total):.2f} |")

    return lines


def markdown_quality_table(total: int, valid_fault_ratio_true: int, minimum_required_true: int) -> list[str]:
    valid_fault_ratio_false = total - valid_fault_ratio_true
    minimum_required_false = total - minimum_required_true

    lines: list[str] = []
    lines.append("| metric | count | ratio_percent |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| has_valid_fault_ratio_true | {valid_fault_ratio_true:,} | {percent(valid_fault_ratio_true, total):.2f} |"
    )
    lines.append(
        f"| has_valid_fault_ratio_false | {valid_fault_ratio_false:,} | {percent(valid_fault_ratio_false, total):.2f} |"
    )
    lines.append(
        f"| has_minimum_required_fields_true | {minimum_required_true:,} | {percent(minimum_required_true, total):.2f} |"
    )
    lines.append(
        f"| has_minimum_required_fields_false | {minimum_required_false:,} | {percent(minimum_required_false, total):.2f} |"
    )
    return lines


def build_markdown_report(
    inspections: list[dict[str, Any]],
    case_code_counter: Counter[str],
    road_type_counter: Counter[str],
    excluded_by_reason: Counter[str],
    rare_threshold: int,
    top_n: int,
) -> str:
    total_inspected = len(inspections)
    valid_fault_ratio_true = sum(1 for item in inspections if item["has_valid_fault_ratio"])
    minimum_required_true = sum(1 for item in inspections if item["has_minimum_required_fields"])

    rare_case_counter = Counter(
        {
            code: count
            for code, count in case_code_counter.items()
            if count < rare_threshold
        }
    )

    top_case_rows = build_rank_rows(case_code_counter, total_inspected, top_n=top_n)
    top_road_rows = build_rank_rows(road_type_counter, total_inspected, top_n=top_n)
    rare_case_rows = build_rank_rows(rare_case_counter, total_inspected, top_n=None)

    top_case_total = sum(row["count"] for row in top_case_rows)
    top_road_total = sum(row["count"] for row in top_road_rows)

    lines: list[str] = []
    lines.append("# 팀 공유용 EDA 보고서")
    lines.append("")
    lines.append("이 문서는 `label_inspection_cache.json` 및 `subset_pipeline_summary.json` 기준으로 생성한 팀 공유용 Markdown 보고서입니다.")
    lines.append("")

    lines.append("## 1. 전체 개요")
    lines.append("")
    lines.append(f"- 전체 검사 라벨 수: **{total_inspected:,}**")
    lines.append(f"- 고유 case_code 수: **{len(case_code_counter):,}**")
    lines.append(f"- 고유 road_type 수: **{len(road_type_counter):,}**")
    lines.append(f"- 희귀 case_code 기준: **count < {rare_threshold}**")
    lines.append(f"- 희귀 case_code 종류 수: **{len(rare_case_counter):,}**")
    lines.append(f"- 희귀 case_code 전체 건수: **{sum(rare_case_counter.values()):,}**")
    lines.append("")

    lines.append("## 2. 데이터 품질 요약")
    lines.append("")
    lines.extend(markdown_quality_table(total_inspected, valid_fault_ratio_true, minimum_required_true))
    lines.append("")

    lines.append("## 3. 발표용 핵심 포인트")
    lines.append("")
    if top_case_rows:
        first = top_case_rows[0]
        lines.append(
            f"- 가장 많은 case_code는 **{first['key']}**이며, **{first['count']:,}건**으로 전체의 **{first['ratio_percent']:.2f}%**입니다."
        )
    if top_road_rows:
        first = top_road_rows[0]
        lines.append(
            f"- 가장 많은 road_type은 **{first['key']}**이며, **{first['count']:,}건**으로 전체의 **{first['ratio_percent']:.2f}%**입니다."
        )
    lines.append(
        f"- 상위 {top_n}개 case_code 누적 비율은 **{percent(top_case_total, total_inspected):.2f}%**입니다."
    )
    lines.append(
        f"- 상위 {top_n}개 road_type 누적 비율은 **{percent(top_road_total, total_inspected):.2f}%**입니다."
    )
    if excluded_by_reason:
        top_reason, top_reason_count = excluded_by_reason.most_common(1)[0]
        lines.append(
            f"- 주요 제외 사유는 **{top_reason}**이며, **{top_reason_count:,}건**입니다."
        )
    lines.append("")

    lines.append(f"## 4. 상위 case_code TOP {top_n}")
    lines.append("")
    lines.extend(markdown_table(top_case_rows, "case_code"))
    lines.append("")

    lines.append(f"## 5. 상위 road_type TOP {top_n}")
    lines.append("")
    lines.extend(markdown_table(top_road_rows, "road_type"))
    lines.append("")

    lines.append("## 6. 희귀 case_code 전체 목록")
    lines.append("")
    lines.extend(markdown_table(rare_case_rows, "case_code"))
    lines.append("")

    lines.append("## 7. 제외 사유")
    lines.append("")
    lines.extend(markdown_reason_table(excluded_by_reason, total_inspected))
    lines.append("")

    lines.append("## 8. 해석 가이드")
    lines.append("")
    lines.append("- `case_code`와 `road_type`은 데이터셋 키값을 그대로 사용했습니다.")
    lines.append("- `ratio_percent`는 전체 검사 라벨 수 대비 비율입니다.")
    lines.append("- `cumulative_ratio_percent`는 상위 항목 누적 비율입니다.")
    lines.append("- 희귀 case_code는 기본적으로 `count < rare_threshold` 기준으로 계산합니다.")
    lines.append("- 이 문서는 팀 공유용 요약본이며, 모델링/샘플링 판단 시 데이터 불균형 확인 용도로 사용할 수 있습니다.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a single Markdown EDA report from cache")
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-file-name", type=str, default=DEFAULT_OUTPUT_FILE_NAME)
    parser.add_argument("--rare-threshold", type=int, default=DEFAULT_RARE_THRESHOLD)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args()

    inspections = load_label_inspection_entries(args.cache_path)
    excluded_by_reason = extract_excluded_by_reason(args.summary_path)

    case_code_counter = Counter(
        item["case_code"]
        for item in inspections
        if item["case_code"] is not None
    )
    road_type_counter = Counter(
        item["road_type"]
        for item in inspections
        if item["road_type"] is not None
    )

    report_markdown = build_markdown_report(
        inspections=inspections,
        case_code_counter=case_code_counter,
        road_type_counter=road_type_counter,
        excluded_by_reason=excluded_by_reason,
        rare_threshold=args.rare_threshold,
        top_n=args.top_n,
    )

    ensure_dir(args.output_dir)
    output_path = args.output_dir / args.output_file_name
    with output_path.open("w", encoding="utf-8") as file:
        file.write(report_markdown)

    print("[EDA] Markdown 보고서 생성 완료")
    print(f"[EDA] output_path={output_path}")
    print(f"[EDA] total_inspected_items={len(inspections)}")
    print(f"[EDA] case_code_unique_count={len(case_code_counter)}")
    print(f"[EDA] road_type_unique_count={len(road_type_counter)}")
    print(f"[EDA] rare_threshold={args.rare_threshold}")
    print(f"[EDA] top_n={args.top_n}")


if __name__ == "__main__":
    main()
