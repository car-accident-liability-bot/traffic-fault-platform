from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 주석:
# - training-runner만 먼저 설치했을 때도 같은 저장소 안의 ai-core 소스를 우선 찾도록
#   로컬 모노레포 경로를 보수적으로 sys.path에 추가한다.
# - 이렇게 하면 repo 내부 실행 경로에서는 콘솔 엔트리포인트가 바로 깨지는 문제를 줄일 수 있다.
CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[5]
AI_CORE_SRC = REPO_ROOT / "packages" / "ai-core" / "src"
if AI_CORE_SRC.is_dir():
    ai_core_src_text = str(AI_CORE_SRC)
    if ai_core_src_text not in sys.path:
        sys.path.insert(0, ai_core_src_text)

from traffic_ai_core.data.zip_subset import (
    DEFAULT_DATASET_BASE_URL,
    LabelQualityOptions,
    build_all_category_records,
    derive_dataset_metadata,
    extract_selected_files,
    preprocess_category_records,
    sample_per_category,
    write_manifest_csv,
    write_manifest_json,
    write_summary_json,
)


DEFAULT_MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
DEFAULT_WORKING_DIR = REPO_ROOT / "data" / "working" / "pipeline_subset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Training 영상 ZIP에서 카테고리별 subset manifest를 생성하고, "
            "필요하면 선택 샘플만 working 디렉터리로 추출합니다."
        )
    )

    raw_source_group = parser.add_mutually_exclusive_group(required=True)
    raw_source_group.add_argument(
        "--raw-dir",
        type=Path,
        help="실행 시 사용할 raw 영상 ZIP 디렉터리 경로",
    )
    raw_source_group.add_argument(
        "--raw-url",
        type=str,
        help="실행 시 사용할 raw 영상 ZIP 디렉터리 공용 URL",
    )

    label_source_group = parser.add_mutually_exclusive_group(required=True)
    label_source_group.add_argument(
        "--label-dir",
        type=Path,
        help="실행 시 사용할 label JSON ZIP 디렉터리 경로",
    )
    label_source_group.add_argument(
        "--label-url",
        type=str,
        help="실행 시 사용할 label JSON ZIP 디렉터리 공용 URL",
    )

    parser.add_argument(
        "--dataset-base-url",
        type=str,
        default=DEFAULT_DATASET_BASE_URL,
        help="manifest와 summary에 기록할 공용 데이터셋 기준 URL",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default=None,
        help="manifest에 기록할 split 이름. 미지정 시 경로 또는 URL에서 자동 추론",
    )
    parser.add_argument(
        "--raw-dir-name",
        type=str,
        default=None,
        help="manifest에 기록할 raw 디렉터리 이름. 미지정 시 입력 경로 또는 URL에서 자동 추론",
    )
    parser.add_argument(
        "--label-dir-name",
        type=str,
        default=None,
        help="manifest에 기록할 label 디렉터리 이름. 미지정 시 입력 경로 또는 URL에서 자동 추론",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=10,
        help="카테고리별 샘플 수. 기본값은 10",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="재현 가능한 샘플링용 랜덤 시드",
    )
    parser.add_argument(
        "--manifest-csv-out",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "subset_pipeline.csv",
        help="manifest CSV 출력 경로. 기본값은 저장소 루트 data/manifests 기준",
    )
    parser.add_argument(
        "--manifest-json-out",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "subset_pipeline.json",
        help="manifest JSON 출력 경로. 기본값은 저장소 루트 data/manifests 기준",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "subset_pipeline_summary.json",
        help="summary JSON 출력 경로. 기본값은 저장소 루트 data/manifests 기준",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=DEFAULT_WORKING_DIR,
        help="선택 샘플을 추출할 working 디렉터리. 기본값은 저장소 루트 data/working 기준",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="실제 파일 추출은 생략하고 manifest/summary만 생성",
    )
    parser.add_argument(
        "--disable-label-quality-filter",
        action="store_true",
        help="라벨 JSON 품질 전처리를 끄고 basename 매칭 결과를 그대로 사용",
    )
    parser.add_argument(
        "--rare-case-code-min-frequency",
        type=int,
        default=2,
        help=(
            "희귀 case code를 제외할 최소 빈도 기준. 기본값은 2이며, 1이면 희귀 코드 제외를 끈다"
        ),
    )
    case_code_group = parser.add_mutually_exclusive_group()
    case_code_group.add_argument(
        "--allow-missing-case-code",
        action="store_true",
        help="기본 전처리에서 case code 누락 또는 비정상 샘플 제외를 끄고 유지",
    )
    case_code_group.add_argument(
        "--require-case-code",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    using_url_mode = args.raw_url is not None
    if using_url_mode != (args.label_url is not None):
        raise ValueError("URL 모드를 사용하려면 --raw-url 과 --label-url 을 함께 지정해야 합니다.")

    metadata = derive_dataset_metadata(
        raw_dir=args.raw_dir,
        label_dir=args.label_dir,
        raw_url=args.raw_url,
        label_url=args.label_url,
        dataset_base_url=args.dataset_base_url,
        split_name=args.split_name,
        raw_dir_name=args.raw_dir_name,
        label_dir_name=args.label_dir_name,
    )

    source_description = "공용 URL 인덱스" if using_url_mode else "로컬 디렉터리"
    print(f"[1/5] {source_description}에서 ZIP 내부 확장자와 basename overlap 기준으로 raw/label ZIP을 매칭합니다...")
    category_records, diagnostics = build_all_category_records(
        metadata=metadata,
        raw_dir=args.raw_dir.expanduser().resolve() if args.raw_dir is not None else None,
        label_dir=args.label_dir.expanduser().resolve() if args.label_dir is not None else None,
        raw_url=args.raw_url,
        label_url=args.label_url,
    )

    print("[2/5] 카테고리별 매칭 통계를 출력합니다...")
    for stats in diagnostics:
        print(
            f"  - {stats['category']}: "
            f"raw_zip={stats['raw_zip_name']}, "
            f"label_zip={stats['label_zip_name']}, "
            f"raw={stats['raw_count']}, "
            f"label={stats['label_count']}, "
            f"matched={stats['matched_count']}, "
            f"raw_only={stats['raw_only_count']}, "
            f"label_only={stats['label_only_count']}"
        )

    print("[3/5] 라벨 JSON 품질 기준으로 subset 후보를 전처리합니다...")
    filtered_category_records, preprocessing_summary = preprocess_category_records(
        category_records=category_records,
        options=LabelQualityOptions(
            enabled=not args.disable_label_quality_filter,
            require_fault_ratio=True,
            require_road_type=True,
            require_case_code=not args.allow_missing_case_code,
            rare_case_code_min_frequency=max(1, args.rare_case_code_min_frequency),
        ),
    )
    print(
        f"  - 전처리 전/후: {preprocessing_summary['total_before_filter']} -> "
        f"{preprocessing_summary['total_after_filter']}"
    )
    if preprocessing_summary.get("excluded_by_reason"):
        for reason, count in preprocessing_summary["excluded_by_reason"].items():
            print(f"    * 제외 사유 {reason}: {count}")

    print(f"[4/5] 카테고리별 {args.per_category}개씩 샘플링합니다...")
    selected_records = sample_per_category(
        category_records=filtered_category_records,
        per_category=args.per_category,
        seed=args.seed,
    )

    write_manifest_csv(selected_records, args.manifest_csv_out)
    write_manifest_json(selected_records, args.manifest_json_out)
    write_summary_json(
        output_path=args.summary_out,
        selected_records=selected_records,
        diagnostics=diagnostics,
        metadata=metadata,
        per_category=args.per_category,
        seed=args.seed,
        preprocessing_summary=preprocessing_summary,
    )

    print(f"  - manifest CSV 저장 완료 : {args.manifest_csv_out}")
    print(f"  - manifest JSON 저장 완료: {args.manifest_json_out}")
    print(f"  - summary 저장 완료      : {args.summary_out}")
    print(f"  - 총 선택 샘플 수       : {len(selected_records)}")

    if args.skip_extract:
        print("[5/5] 추출 단계는 생략했습니다.")
        return

    print("[5/5] 선택된 샘플만 working 디렉터리로 추출합니다...")
    extract_selected_files(selected_records, args.extract_dir)
    print(f"  - 추출 완료: {args.extract_dir}")


if __name__ == "__main__":
    main()
