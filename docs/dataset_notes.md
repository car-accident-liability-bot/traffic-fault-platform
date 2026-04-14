# Traffic Dataset Notes

## Overview

이 문서는 `https://data.taeo-dev.com/dataset/traffic` 데이터셋을 기준으로,
차대차 사고 영상 기반 멀티모달 챗봇 프로젝트의 subset 전략을 정리한 문서다.

초기 범위에서는 **영상 ZIP + JSON 라벨 ZIP**만 사용한다.
이미지 ZIP은 초기 파이프라인 범위에서 제외한다.

---

## Public Dataset Structure

```text
traffic/
  1.Training/
    label_data_231108_add/
    raw_data_231108_add/
  2.Validation/
    label_data_231108_add/
    raw_data_231108_add/
```

공용 URL 기준:
- `https://data.taeo-dev.com/dataset/traffic`

---

## Confirmed Usage Policy

### 사용 대상
- `TS_차대차_영상_*.zip`
- `TL_차대차_영상_*.zip`

### 제외 대상
- `TS_차대차_이미지_*.zip`
- `TL_차대차_이미지_*.zip`

### split 정책
- `1.Training`: subset 생성 대상
- `2.Validation`: 최종 홀드아웃 유지

---

## Subset Strategy

### Pipeline Validation Subset

목적:
- ZIP 인덱싱 검증
- 영상/JSON basename 1:1 매칭 검증
- working 추출 검증
- 후속 학습/추론 파이프라인 입력 검증

기준:
- 카테고리별 10개
- 총 약 80개 샘플

---

## Matching Rule

영상과 JSON은 **basename 기준 1:1 매칭**한다.

예시:
- `bb_1_150113_vehicle_121_037.mp4`
- `bb_1_150113_vehicle_121_037.json`

---

## Manifest Policy

manifest에는 팀 공용 기준 정보만 저장한다.

저장 항목 예:
- split
- category
- sample_id
- raw_zip_relative_path
- label_zip_relative_path
- raw_zip_public_url
- label_zip_public_url
- raw_zip_name
- label_zip_name
- video_member_name
- label_member_name

저장하지 않는 항목:
- 개인 NAS 절대경로
- 로컬 머신 전용 경로

---

## Execution Policy

기본 실행은 **공용 URL 디렉터리 기준**으로 처리한다.

즉:
- 문서와 manifest는 공용 URL / 상대경로 중심
- 실제 ZIP 읽기도 `--raw-url`, `--label-url` 기준
- 전체 ZIP을 로컬로 복사하지 않고 원격 ZIP을 직접 읽는다.

필요하면 fallback으로 아래도 허용한다.
- `--raw-dir`
- `--label-dir`

---

## Subset Preprocessing Policy

subset 후보를 만들 때 아래 라벨 품질 기준을 기본 적용한다.

- 과실비율 필드가 비어 있거나 `NaN`, 공백, `-`, `null`, `None` 등 placeholder 성격의 값인 JSON 제외
- 도로유형 필드가 미상/불명/기타 등으로 해석되는 JSON 제외
- case code가 누락되었거나 비정상으로 해석되는 JSON 제외
- 핵심 라벨 필드가 사실상 비어 있는 JSON 제외
- case code가 전체 후보에서 지나치게 드문 경우 제외

주의:
- 로컬 절대경로는 manifest에 저장하지 않는다.
- URL 모드에서는 디렉터리 인덱스와 HTTP byte-range 요청이 가능한 서버를 전제로 한다.

---

## Command Example Policy

문서에는 실제 공용 URL 기준 복붙용 실행 예시를 둔다.

1. 실제 공용 URL 기준 복붙용 예시
   - 예: `https://data.taeo-dev.com/dataset/traffic/1.Training/raw_data_231108_add/`

로컬 절대경로(`/volume1/...`)는 저장소 기준 문서에 고정하지 않는다.


팀원 공유용으로 정리하면 아래처럼 보면 됩니다.

**실행 명령어**

```bash
# URL 모드
python -m training_runner.cli.build_pipeline_subset \
  --raw-url "RAW_ZIP_디렉터리_URL" \
  --label-url "LABEL_ZIP_디렉터리_URL"

# 로컬 경로 모드
python -m training_runner.cli.build_pipeline_subset \
  --raw-dir "RAW_ZIP_디렉터리_경로" \
  --label-dir "LABEL_ZIP_디렉터리_경로"
```

**옵션별 정리**

* `--raw-dir` : **필수(로컬 모드일 때)** / raw 영상 ZIP 로컬 디렉터리 경로를 지정합니다.

* `--raw-url` : **필수(URL 모드일 때)** / raw 영상 ZIP 공용 URL 디렉터리를 지정합니다.

* `--label-dir` : **필수(로컬 모드일 때)** / label JSON ZIP 로컬 디렉터리 경로를 지정합니다.

* `--label-url` : **필수(URL 모드일 때)** / label JSON ZIP 공용 URL 디렉터리를 지정합니다.

* `--dataset-base-url` : **선택** / 기본값은 `https://data.taeo-dev.com/dataset/traffic` / manifest와 summary에 기록할 기준 데이터셋 URL입니다.

* `--split-name` : **선택** / 기본값은 `자동 추론` / 결과 메타데이터에 들어갈 split 이름입니다.

* `--raw-dir-name` : **선택** / 기본값은 `자동 추론` / 결과 메타데이터에 들어갈 raw 디렉터리 이름입니다.

* `--label-dir-name` : **선택** / 기본값은 `자동 추론` / 결과 메타데이터에 들어갈 label 디렉터리 이름입니다.

* `--per-category` : **선택** / 기본값은 `10` / 카테고리별 샘플 수를 지정합니다.

* `--seed` : **선택** / 기본값은 `42` / 재현 가능한 샘플링용 랜덤 시드입니다.

* `--manifest-csv-out` : **선택** / 기본값은 `data/manifests/subset_pipeline.csv` / subset 결과 CSV manifest 저장 경로입니다.

* `--manifest-json-out` : **선택** / 기본값은 `data/manifests/subset_pipeline.json` / subset 결과 JSON manifest 저장 경로입니다.

* `--summary-out` : **선택** / 기본값은 `data/manifests/subset_pipeline_summary.json` / subset summary JSON 저장 경로입니다.

* `--extract-dir` : **선택** / 기본값은 `data/working/pipeline_subset` / 선택된 샘플을 실제로 추출할 working 디렉터리입니다.

* `--skip-extract` : **선택** / 기본값은 `미사용(False)` / 실제 파일 추출 없이 manifest와 summary만 생성합니다.

* `--disable-label-quality-filter` : **선택** / 기본값은 `미사용(False)` / 라벨 품질 전처리를 끄고 basename 매칭 결과를 그대로 사용합니다.

* `--rare-case-code-min-frequency` : **선택** / 기본값은 `2` / 희귀 case code를 제외할 최소 빈도 기준입니다.

* `--allow-missing-case-code` : **선택** / 기본값은 `미사용(False)` / case code가 없거나 비정상인 샘플도 제외하지 않고 유지합니다.

**주의사항**

* 로컬 모드면 `--raw-dir`, `--label-dir` 둘 다 넣어야 합니다.
* URL 모드면 `--raw-url`, `--label-url` 둘 다 넣어야 합니다.
* `raw`는 URL인데 `label`은 dir처럼 섞어서 쓰면 안 됩니다.
* `--split-name`, `--raw-dir-name`, `--label-dir-name`을 안 주면 입력 경로 또는 URL 기준으로 자동 추론됩니다.
* `--skip-extract`를 안 쓰면 manifest 생성 후 실제 파일 추출까지 수행됩니다.
* 기본 전처리에서는 `fault_ratio`, `road_type`, `case_code`가 필요한 샘플만 유지됩니다.

````

**자주 쓰는 전체 예시(URL 모드)**
```bash
python -m training_runner.cli.build_pipeline_subset \
  --raw-url "https://data.taeo-dev.com/dataset/traffic/1.Training/raw_data_231108_add/" \
  --label-url "https://data.taeo-dev.com/dataset/traffic/1.Training/label_data_231108_add/" \
  --dataset-base-url "https://data.taeo-dev.com/dataset/traffic" \
  --split-name "1.Training" \
  --raw-dir-name "raw_data_231108_add" \
  --label-dir-name "label_data_231108_add" \
  --per-category 10 \
  --seed 42 \
  --manifest-csv-out "data/manifests/subset_pipeline.csv" \
  --manifest-json-out "data/manifests/subset_pipeline.json" \
  --summary-out "data/manifests/subset_pipeline_summary.json" \
  --extract-dir "data/working/pipeline_subset"
````

