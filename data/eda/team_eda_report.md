# 팀 공유용 EDA 보고서

이 문서는 `label_inspection_cache.json` 및 `subset_pipeline_summary.json` 기준으로 생성한 팀 공유용 Markdown 보고서입니다.

## 1. 전체 개요

- 전체 검사 라벨 수: **14,099**
- 고유 case_code 수: **126**
- 고유 road_type 수: **78**
- 희귀 case_code 기준: **count < 5**
- 희귀 case_code 종류 수: **25**
- 희귀 case_code 전체 건수: **60**

## 2. 데이터 품질 요약

| metric | count | ratio_percent |
|---|---:|---:|
| has_valid_fault_ratio_true | 13,166 | 93.38 |
| has_valid_fault_ratio_false | 933 | 6.62 |
| has_minimum_required_fields_true | 14,099 | 100.00 |
| has_minimum_required_fields_false | 0 | 0.00 |

## 3. 발표용 핵심 포인트

- 가장 많은 case_code는 **11**이며, **2,621건**으로 전체의 **18.59%**입니다.
- 가장 많은 road_type은 **accident_place:0|accident_place_feature:6**이며, **3,735건**으로 전체의 **26.49%**입니다.
- 상위 10개 case_code 누적 비율은 **52.86%**입니다.
- 상위 10개 road_type 누적 비율은 **75.37%**입니다.
- 주요 제외 사유는 **missing_or_invalid_case_code**이며, **1,040건**입니다.

## 4. 상위 case_code TOP 10

| rank | case_code | count | ratio_percent | cumulative_ratio_percent |
|---:|---|---:|---:|---:|
| 1 | 11 | 2,621 | 18.59 | 18.59 |
| 2 | 2 | 1,282 | 9.09 | 27.68 |
| 3 | 1 | 868 | 6.16 | 33.84 |
| 4 | 13 | 592 | 4.20 | 38.04 |
| 5 | 194 | 412 | 2.92 | 40.96 |
| 6 | 21 | 380 | 2.70 | 43.66 |
| 7 | 60 | 365 | 2.59 | 46.24 |
| 8 | 126 | 321 | 2.28 | 48.52 |
| 9 | 189 | 309 | 2.19 | 50.71 |
| 10 | 5 | 303 | 2.15 | 52.86 |

## 5. 상위 road_type TOP 10

| rank | road_type | count | ratio_percent | cumulative_ratio_percent |
|---:|---|---:|---:|---:|
| 1 | accident_place:0|accident_place_feature:6 | 3,735 | 26.49 | 26.49 |
| 2 | accident_place:0|accident_place_feature:0 | 2,437 | 17.28 | 43.78 |
| 3 | accident_place:1|accident_place_feature:10 | 884 | 6.27 | 50.05 |
| 4 | accident_place:2|accident_place_feature:18 | 685 | 4.86 | 54.90 |
| 5 | accident_place:3|accident_place_feature:10 | 587 | 4.16 | 59.07 |
| 6 | accident_place:5|accident_place_feature:22 | 578 | 4.10 | 63.17 |
| 7 | accident_place:3|accident_place_feature:11 | 482 | 3.42 | 66.59 |
| 8 | accident_place:13|accident_place_feature:41 | 458 | 3.25 | 69.83 |
| 9 | accident_place:2|accident_place_feature:49 | 442 | 3.13 | 72.97 |
| 10 | accident_place:13|accident_place_feature:38 | 338 | 2.40 | 75.37 |

## 6. 희귀 case_code 전체 목록

| rank | case_code | count | ratio_percent | cumulative_ratio_percent |
|---:|---|---:|---:|---:|
| 1 | 52 | 4 | 0.03 | 0.03 |
| 2 | 86 | 4 | 0.03 | 0.06 |
| 3 | 67 | 4 | 0.03 | 0.09 |
| 4 | 80 | 4 | 0.03 | 0.11 |
| 5 | 68 | 4 | 0.03 | 0.14 |
| 6 | 112 | 4 | 0.03 | 0.17 |
| 7 | 88 | 3 | 0.02 | 0.19 |
| 8 | 195 | 3 | 0.02 | 0.21 |
| 9 | 74 | 3 | 0.02 | 0.23 |
| 10 | 55 | 3 | 0.02 | 0.26 |
| 11 | 120 | 3 | 0.02 | 0.28 |
| 12 | 58 | 3 | 0.02 | 0.30 |
| 13 | 36 | 2 | 0.01 | 0.31 |
| 14 | -1 | 2 | 0.01 | 0.33 |
| 15 | 43 | 2 | 0.01 | 0.34 |
| 16 | 75 | 2 | 0.01 | 0.35 |
| 17 | 15 | 2 | 0.01 | 0.37 |
| 18 | 41 | 1 | 0.01 | 0.38 |
| 19 | 199 | 1 | 0.01 | 0.38 |
| 20 | 114 | 1 | 0.01 | 0.39 |
| 21 | 18 | 1 | 0.01 | 0.40 |
| 22 | 364 | 1 | 0.01 | 0.40 |
| 23 | 215 | 1 | 0.01 | 0.41 |
| 24 | 17 | 1 | 0.01 | 0.42 |
| 25 | 72 | 1 | 0.01 | 0.43 |

## 7. 제외 사유

| rank | reason | count | ratio_percent |
|---:|---|---:|---:|
| 1 | missing_or_invalid_case_code | 1,040 | 7.38 |
| 2 | missing_or_invalid_fault_ratio | 933 | 6.62 |
| 3 | rare_case_code | 7 | 0.05 |

## 8. 해석 가이드

- `case_code`와 `road_type`은 데이터셋 키값을 그대로 사용했습니다.
- `ratio_percent`는 전체 검사 라벨 수 대비 비율입니다.
- `cumulative_ratio_percent`는 상위 항목 누적 비율입니다.
- 희귀 case_code는 기본적으로 `count < rare_threshold` 기준으로 계산합니다.
- 이 문서는 팀 공유용 요약본이며, 모델링/샘플링 판단 시 데이터 불균형 확인 용도로 사용할 수 있습니다.

## 9. 실행 명령어

-상위 개수 변경 `python scripts/eda/eda_from_label_cache.py --top-n 15`
-희귀 기준 변경 `python scripts/eda/eda_from_label_cache.py --rare-threshold 10`
