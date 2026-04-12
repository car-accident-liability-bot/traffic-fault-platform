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
