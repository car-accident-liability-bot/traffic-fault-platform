# docs/dataset_notes.md

# Traffic Dataset Notes

## Overview

이 문서는 `https://data.taeo-dev.com/dataset/traffic` 데이터셋을 기반으로,
차대차 사고 영상 멀티모달 챗봇 프로젝트에서 사용할 영상 서브셋 전략을 정리한 문서다.

본 프로젝트의 1차 목표는 다음과 같다.

- 차대차 사고 영상을 입력으로 받는다.
- 영상 내용을 바탕으로 사고 상황을 이해한다.
- 과실비율 판단에 도움이 되는 근거를 추출한다.
- 사용자에게 설명 가능한 형태의 응답을 제공한다.

초기 범위에서는 AI 파이프라인 검증 및 MVP 구축에 집중한다.

---

## Dataset Location

### Public URL
- `https://data.taeo-dev.com/dataset/traffic`

### Directory Structure
```text
traffic/
  1.Training/
    label_data_231108_add/
    raw_data_231108_add/
  2.Validation/
    label_data_231108_add/
    raw_data_231108_add/
```

---

## Confirmed Facts

### Training
- `1.Training/raw_data_231108_add`: 약 2.7TB
- `1.Training/label_data_231108_add`: 약 2.8GB

### Validation
- `2.Validation/raw_data_231108_add`: 약 341GB
- `2.Validation/label_data_231108_add`: 약 359MB

### ZIP 구성
각 split에는 총 16개의 ZIP이 있으며, 다음 두 종류가 섞여 있다.

- `TS_차대차_영상_*.zip` / `TL_차대차_영상_*.zip`
- `TS_차대차_이미지_*.zip` / `TL_차대차_이미지_*.zip`

### 1차 프로젝트 사용 범위
초기 프로젝트에서는 아래만 사용한다.

- `TS_차대차_영상_*.zip`
- `TL_차대차_영상_*.zip`

이미지 ZIP은 초기 범위에서 제외한다.

---

## Video Dataset Facts

현재 확인된 Training 기준 영상 샘플 수는 다음과 같다.

- 영상 파일 수: **14,099개**
- JSON 파일 수: **14,099개**

또한 영상 파일과 JSON 파일은 basename 기준 1:1 매칭 구조를 가진다.

예시:
- `bb_1_150113_vehicle_121_037.mp4`
- `bb_1_150113_vehicle_121_037.json`

---

## Category Groups

현재 영상 ZIP은 대략 다음 8개 카테고리로 구성된다.

- 회전교차로
- 차도와 차도가 아닌 장소
- 주차장(또는 차도가 아닌 장소)
- 고속도로(자동차전용도로 포함)
- T자형교차로
- 사거리교차로(신호등 없음)
- 사거리교차로(신호등 있음)
- 직선도로

샘플링 시 용량 비례가 아니라 카테고리 균형 기준으로 선택한다.

---

## Validation Policy

`2.Validation`은 최종 홀드아웃 검증셋으로 유지한다.

원칙:
- Validation 데이터는 서브셋 생성 대상이 아니다.
- Validation 데이터는 최종 점검 단계에서만 사용한다.
- 실제 subset 생성은 `1.Training` 내부에서만 수행한다.

---

## Subset Strategy

### 1) Pipeline Validation Subset
목적:
- ZIP 인덱싱 검증
- 영상/JSON 1:1 매칭 검증
- 프레임 샘플링 검증
- 추론 입력 포맷 검증
- 응답 구조 검증

기준:
- 카테고리별 10개
- 총 80개 샘플

### 2) MVP Subset
목적:
- 데모 품질 확보
- 프롬프트 조정
- 추론 로직 검증
- 오류 유형 분석

- 카테고리별 70~100개
- 총 600~800개 샘플

---

## Sampling Policy

### Rules
- 용량 비례 샘플링은 사용하지 않는다.
- 카테고리 균형 샘플링을 사용한다.
- 영상과 JSON은 basename 기준으로 매칭한다.
- 초기에는 원본 mp4 1개를 샘플 1개로 사용한다.
- 필요 시에만 후속 단계에서 재클립을 검토한다.

---

## Storage Policy

### Do Not
- 원본 ZIP을 subset별로 복사하지 않는다.
- 전체 ZIP을 한 번에 다 풀지 않는다.
- 전체 프레임을 미리 추출하지 않는다.

### Do
- subset은 manifest 기반으로 관리한다.
- 실제 추출은 필요한 샘플만 한다.
- working 디렉터리에는 현재 실험에 필요한 샘플만 둔다.

---

## Manifest Policy

예시:
```text
data/manifests/
  subset_pipeline.csv
  subset_pipeline_summary.json
  subset_mvp.csv
  subset_expand.csv
```

장점:
- 원본 복사 없이 subset을 관리할 수 있다.
- 용량 증가가 거의 없다.
- 샘플링 기준을 바꿔도 재복사가 필요 없다.
- MVP와 확장셋을 동시에 관리하기 쉽다.

---

## Working Directory Policy

파이프라인 검증 단계에서는 선택된 샘플만 실제 작업 폴더로 추출한다.

예시:
```text
data/working/pipeline_subset/
  raw/
    직선도로/
    회전교차로/
    ...
  label/
    직선도로/
    회전교차로/
    ...
```

즉, 전체 ZIP을 푸는 것이 아니라 선택된 샘플만 제한적으로 추출한다.

---

## Path Policy

이 프로젝트는 다음 두 기준을 함께 사용한다.

### Public Reference
- 문서
- 설정
- manifest
- 공유용 경로

사용 값:
- `https://data.taeo-dev.com/dataset/traffic`

### Local Processing
- ZIP 열기
- subset 추출
- working 파일 생성

사용 값:
- `/volume1/project/dataset/traffic`

즉,
- 표현과 공유는 URL 기준
- 실제 처리와 실행은 로컬 경로 기준

---

## Initial Implementation Scope

초기 구현에서는 아래를 우선 수행한다.

1. Training 영상 ZIP 인덱싱
2. 영상/JSON basename 매칭
3. 카테고리별 10개 샘플링
4. `subset_pipeline.csv` 생성
5. 선택된 80쌍 working 추출
6. 파이프라인 검증

---

## Summary

현재 프로젝트의 핵심 원칙은 다음과 같다.

- 전체 3TB를 그대로 사용하는 것이 아니라 영상만 우선 사용한다.
- Training에서만 subset을 생성한다.
- Validation은 최종 홀드아웃으로 유지한다.
- 파이프라인 검증용으로 카테고리별 10개씩 샘플링한다.
- subset은 폴더 복사형이 아니라 manifest 기반으로 관리한다.
- 실제 working 폴더에는 필요한 샘플만 추출한다.