# README.md

# traffic-fault-platform

차대차 사고 영상 기반 멀티모달 챗봇 프로젝트를 위한 모노레포다.

초기 목표는 다음과 같다.

- 차대차 사고 영상을 입력으로 받는다.
- 사고 장면을 이해한다.
- 과실비율 판단에 도움이 되는 근거를 추출한다.
- 사용자에게 설명 가능한 형태로 응답한다.

---

## Repository Structure

```text
traffic-fault-platform/
├─ README.md
├─ .gitignore
├─ configs/
│  ├─ dataset/
│  ├─ model/
│  └─ prompt/
├─ docs/
│  └─ dataset_notes.md
├─ data/
│  ├─ manifests/
│  │  └─ .gitkeep
│  └─ working/
│     └─ .gitkeep
├─ artifacts/
│  └─ .gitkeep
├─ checkpoints/
│  └─ .gitkeep
├─ outputs/
│  └─ .gitkeep
├─ apps/
│  ├─ training-runner/
│  │  └─ scripts/
│  ├─ inference-service/
│  │  └─ app/
│  │     └─ routes/
│  ├─ backend-api-java/
│  │  └─ src/
│  │     └─ main/
│  │        ├─ java/
│  │        └─ resources/
│  └─ frontend-react/
│     └─ src/
├─ packages/
│  ├─ ai-core/
│  │  └─ src/
│  │     └─ traffic_ai_core/
│  │        ├─ data/
│  │        ├─ schemas/
│  │        ├─ video/
│  │        ├─ prompts/
│  │        └─ utils/
│  └─ contracts/
│     ├─ json-schema/
│     ├─ openapi/
│     └─ examples/
└─ scripts/
```

---

## Directory Roles

### `apps/`
실행 가능한 애플리케이션 영역이다.

- `training-runner`
  - 서브셋 생성
  - 학습 실행
  - 오프라인 검증
- `inference-service`
  - 멀티모달 추론 서버
- `backend-api-java`
  - 업로드/저장/추론 요청 중계 API
- `frontend-react`
  - 영상 업로드, 질문 입력, 결과 출력 UI

### `packages/`
공통 모듈 영역이다.

- `ai-core`
  - 학습과 추론이 공통으로 사용하는 Python 로직
- `contracts`
  - JSON schema, OpenAPI, 예제 요청/응답 등 언어 중립 계약

### `configs/`
데이터셋, 모델, 프롬프트 관련 설정 파일을 둔다.

### `docs/`
프로젝트 문서를 둔다.

### `data/`
manifest와 working subset 등 실험용 데이터를 둔다.

### `artifacts/`, `checkpoints/`, `outputs/`
학습 산출물, 체크포인트, 결과물 저장 영역이다.

---

## Monorepo Policy

이 저장소는 모노레포로 운영한다.

원칙:
- 실행 앱은 `apps/` 아래에 둔다.
- 공통 코드는 `packages/` 아래에 둔다.
- Python, Java, React를 같은 저장소 안에서 관리한다.
- Java는 Python 코어를 직접 import 하지 않는다.
- Java와 Python 사이 연결은 API 계약(`packages/contracts`)으로 맞춘다.

---

## Dataset Policy

데이터셋 기준 문서는 `docs/dataset_notes.md`를 따른다.

### Public URL
- `https://data.taeo-dev.com/dataset/traffic`

### Local Path
- `/volume1/project/dataset/traffic`

### Initial Scope
- 1차에서는 영상 ZIP + JSON 라벨 ZIP만 사용한다.
- 이미지 ZIP은 초기 범위에서 제외한다.
- `1.Training`에서만 subset을 생성한다.
- `2.Validation`은 최종 홀드아웃으로 유지한다.

---

## Initial Subset Plan

초기 파이프라인 검증용 subset은 아래 기준으로 생성한다.

- 대상: Training 영상 ZIP
- 방식: 카테고리별 균형 샘플링
- 기준: 카테고리별 10개
- 결과: 총 80개 샘플
- 저장: `data/manifests/subset_pipeline.csv`

실제 working 파일이 필요하면 선택된 샘플만 `data/working/`으로 추출한다.

---

## Path Policy

경로는 다음 두 기준을 구분한다.

### Public Reference
- 문서
- 설정
- manifest
- 공유용 경로

### Local Processing
- ZIP 처리
- subset 추출
- working 파일 생성

즉,
- 설명은 URL 기준
- 실행은 로컬 경로 기준

---

## Development Flow

1. 모노레포 기본 구조 생성
2. `.gitignore` 설정
3. dataset notes 정리
4. pipeline subset 생성
5. JSON 구조 확인
6. 추론 파이프라인 연결
7. MVP subset 생성
8. backend / frontend 통합

---

## Notes

- 원본 ZIP은 복사하지 않는다.
- 전량 압축 해제는 지양한다.
- subset은 manifest 기반으로 관리한다.
- working 디렉터리에는 필요한 샘플만 추출한다.

## Additional Notes

 - 깃 브런치 전략은 의논 후 결정한다.