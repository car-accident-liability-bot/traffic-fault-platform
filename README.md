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
├─ pyproject.toml
├─ apps/
│  ├─ training-runner/
│  │  ├─ pyproject.toml
│  │  ├─ README.md
│  │  ├─ scripts/
│  │  │  └─ build_pipeline_subset.py
│  │  └─ src/
│  │     └─ training_runner/
│  │        ├─ __init__.py
│  │        └─ cli/
│  │           ├─ __init__.py
│  │           └─ build_pipeline_subset.py
│  ├─ inference-service/
│  │  ├─ pyproject.toml
│  │  ├─ README.md
│  │  └─ src/
│  │     └─ traffic_inference_service/
│  │        ├─ __init__.py
│  │        ├─ main.py
│  │        └─ api/
│  │           ├─ __init__.py
│  │           └─ routes.py
│  ├─ backend-api-java/
│  │  └─ README.md
│  └─ frontend-react/
│     └─ README.md
├─ packages/
│  ├─ ai-core/
│  │  ├─ pyproject.toml
│  │  ├─ README.md
│  │  └─ src/
│  │     └─ traffic_ai_core/
│  │        ├─ __init__.py
│  │        └─ data/
│  │           ├─ __init__.py
│  │           └─ zip_subset.py
│  └─ contracts/
│     ├─ README.md
│     ├─ openapi/
│     │  └─ .gitkeep
│     ├─ json-schema/
│     │  └─ .gitkeep
│     └─ examples/
│        └─ .gitkeep
├─ configs/
│  └─ README.md
├─ docs/
│  ├─ dataset_notes.md
│  ├─ project_proposal.md
│  └─ python_setup.md
├─ data/
│  ├─ manifests/
│  │  └─ .gitkeep
│  └─ working/
│     └─ .gitkeep
├─ artifacts/
│  └─ .gitkeep
├─ checkpoints/
│  └─ .gitkeep
└─ outputs/
   └─ .gitkeep
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
  - 업로드, 저장, 추론 요청 중계 API
- `frontend-react`
  - 영상 업로드, 질문 입력, 결과 출력 UI

### `packages/`
공통 모듈 영역이다.

- `ai-core`
  - 학습과 추론이 공통으로 사용하는 Python 로직
- `contracts`
  - JSON Schema, OpenAPI, 예제 요청/응답 등 언어 중립 계약

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

### Initial Scope

- 1차에서는 영상 ZIP과 JSON 라벨 ZIP만 사용한다.
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
- 관리 방식: manifest 기반 관리

실제 working 파일이 필요하면 선택된 샘플만 `data/working/`으로 추출한다.

---

## Path Policy

경로는 아래 두 기준을 구분한다.

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

- 설명은 공용 URL 기준으로 한다.
- 실행은 기본적으로 공용 URL 기준으로 한다.
- 필요할 때만 로컬 경로 fallback을 사용한다.

추가 원칙:

- `/volume1` 같은 개인 NAS 절대경로는 저장소 기준 문서와 코드에 고정하지 않는다.
- manifest에는 로컬 절대경로를 저장하지 않는다.
- 공용 URL, 상대경로, ZIP 이름, member 파일명 중심으로 관리한다.

---

## Development Flow

1. 모노레포 기본 구조 정리
2. `.gitignore` 정리
3. dataset notes 정리
4. pipeline subset 생성
5. manifest 및 summary 구조 검증
6. 추론 파이프라인 연결
7. MVP subset 생성
8. backend 및 frontend 통합

---

## Notes

- 원본 ZIP은 복사하지 않는다.
- 전량 압축 해제는 지양한다.
- subset은 manifest 기반으로 관리한다.
- working 디렉터리에는 필요한 샘플만 추출한다.
- raw/label ZIP 매칭은 ZIP 내부 mp4/json basename overlap 기준으로 처리한다.
- subset 생성은 공용 URL 디렉터리 인덱스와 원격 ZIP 직접 읽기를 기본으로 한다.
- Python 설치 및 실행 방법은 루트 README가 아니라 `docs/python_setup.md`에서 관리한다.
- subset 생성 상세 실행 방법은 `apps/training-runner/README.md`에서 관리한다.
- 깃 브랜치 전략은 추후 의논 후 결정한다.

---

## 커밋 메시지 규칙

우리 팀은 커밋 메시지의 일관성과 가독성을 위해 아래 규칙을 사용한다.

### 기본 형식

```text
type(scope): subject
```

필요한 경우 본문과 푸터를 아래처럼 추가할 수 있다.

```text
type(scope): subject

body

footer
```

### 예시

```text
feat(frontend): NER 추론 결과 출력 영역 추가
fix(backend): metrics_history.json 파싱 오류 수정
refactor(core): 엔티티 매핑 로직 분리
docs(readme): 실행 방법 문서 정리
chore(gitignore): 불필요한 산출물 제외 패턴 추가
```

### type 설명

- `feat`: 새로운 기능 추가
- `fix`: 버그 수정
- `refactor`: 기능 변화 없는 구조 개선
- `docs`: 문서 수정
- `style`: 포맷팅, 세미콜론, 공백 등 비기능 수정
- `test`: 테스트 코드 추가 또는 수정
- `chore`: 빌드, 설정, 패키지, 기타 자잘한 작업
- `perf`: 성능 개선
- `ci`: CI/CD 설정 변경
- `build`: 빌드 시스템 또는 의존성 변경

### scope 예시

- `frontend`
- `backend`
- `core`
- `training`
- `inference`
- `dataset`
- `readme`
- `gitignore`

### subject 작성 규칙

- 너무 길지 않게 작성한다.
- 현재형으로 작성한다.
- 불필요한 마침표는 쓰지 않는다.
- 무엇을 바꿨는지 바로 이해되게 작성한다.

좋은 예:

- `feat(frontend): 추론 결과 차트 영역 추가`
- `fix(training): subject 값 누락 오류 수정`

애매한 예:

- `fix: 수정`
- `feat: 이것저것 변경`

### 권장 사항

- 한 커밋에는 한 가지 목적만 담는다.
- 기능 추가와 리팩터링은 가능하면 분리한다.
- README, 설정 파일, 코드 구조 변경은 되도록 커밋 목적이 드러나게 쓴다.

---

## 참고 문서 및 주요 경로

프로젝트를 볼 때 아래 문서와 경로를 함께 참고한다.

### 문서

- `docs/dataset_notes.md`
  - traffic 데이터셋 구조
  - Training / Validation 사용 정책
  - 영상 ZIP / JSON ZIP 사용 범위
  - basename 기준 1:1 매칭 정책
  - subset 및 manifest 관리 원칙

- `docs/python_setup.md`
  - Python 가상환경 생성 방법
  - `py -3.11` 기준 설치 방법
  - `pip install -e` 실행 순서
  - training-runner, ai-core, inference-service 설치 및 실행 방법

- `apps/training-runner/README.md`
  - subset 생성 앱 설명
  - 실행 진입점 설명
  - CLI 사용 목적 정리

- `packages/ai-core/README.md`
  - 공통 Python 코어 모듈 설명
  - 다른 앱에서 공유하는 역할 정리

- `apps/inference-service/README.md`
  - 추론 서비스 골격 설명
  - 향후 FastAPI 기반 확장 방향 정리

- `configs/README.md`
  - 공용 설정 파일 위치 및 용도 정리

### 주요 데이터 경로

- `data/manifests/`
  - subset 결과 manifest 저장 위치
  - 예:
    - `data/manifests/subset_pipeline.csv`
    - `data/manifests/subset_pipeline.json`
    - `data/manifests/subset_pipeline_summary.json`

- `data/working/`
  - manifest에서 선택된 샘플만 추출하는 working 디렉터리
  - 전체 원본 압축 해제가 아니라 필요한 샘플만 관리하는 용도

- `artifacts/`
  - 모델 아티팩트 저장 경로

- `checkpoints/`
  - 학습 체크포인트 저장 경로

- `outputs/`
  - 추론 결과, 로그성 산출물 등 최종 출력 저장 경로

### 참고 원칙

- 문서는 공용 URL 기준으로 설명한다.
- 실행은 기본적으로 공용 URL 기준으로 하고, 필요할 때만 로컬 경로 fallback을 사용한다.
- manifest에는 로컬 절대경로를 저장하지 않는다.
- 팀 공용 사용성을 위해 개인 NAS 절대경로 하드코딩은 금지한다.