# traffic-ai-core

training-runner와 inference-service가 공통으로 사용하는 Python 코어 모듈이다.

이 모듈은 프로젝트 전반에서 공통으로 필요한 Python 로직을 한곳에 모아두기 위한 패키지다.  
실행 앱마다 동일한 로직을 중복 구현하지 않고, 공통 규칙을 같은 방식으로 재사용하는 것을 목표로 한다.

## 설치

저장소 루트에서 아래처럼 설치한다.

```bash
python -m pip install -e packages/ai-core
```

## 왜 설치가 필요한가

이 모듈은 `src/traffic_ai_core` 패키지를 제공한다.  
따라서 설치하지 않으면 `training-runner`나 `inference-service`에서 공통 코어 모듈을 import할 수 없다.

즉, 아래와 같은 import는 `ai-core`가 먼저 설치되어 있어야 정상 동작한다.

```python
from traffic_ai_core.data.zip_subset import build_all_category_records
```

설치가 되어 있지 않으면 다음과 같은 문제가 생길 수 있다.

- IDE에서 import 해석이 깨진다.
- CLI 실행 시 `ModuleNotFoundError`가 발생할 수 있다.
- 실행 앱이 공통 로직을 참조하지 못해 정상 동작하지 않을 수 있다.

## 이 모듈의 역할

현재 기준으로 `traffic-ai-core`는 아래 역할을 담당한다.

- dataset subset 생성에 필요한 공통 로직 제공
- ZIP 내부 파일 탐색 로직 제공
- raw/label ZIP 매칭 로직 제공
- manifest 생성을 위한 공통 데이터 처리 로직 제공
- training-runner와 inference-service가 함께 사용하는 Python 유틸리티 제공

즉, 실행은 각 앱에서 하더라도 실제 핵심 처리 로직은 이 모듈에 모아두는 구조다.

## 다른 모듈과의 관계

- `apps/training-runner`
  - subset 생성, 학습 실행, 오프라인 검증 등 실행 책임을 가진다.
  - 필요한 공통 처리 로직은 `traffic-ai-core`를 import해서 사용한다.

- `apps/inference-service`
  - 추론 API 서버 역할을 담당한다.
  - 공통 데이터 처리나 코어 유틸리티가 필요할 경우 `traffic-ai-core`를 사용한다.

즉, `traffic-ai-core`는 **실행 앱이 아니라 공통 라이브러리**이고,  
실제 엔트리포인트는 각 `apps/*` 쪽에서 관리한다.

## 구조 설명

이 모듈은 아래 패키지 경로를 기준으로 제공된다.

```text
packages/ai-core/
├─ pyproject.toml
├─ README.md
└─ src/
   └─ traffic_ai_core/
      ├─ __init__.py
      └─ data/
         ├─ __init__.py
         └─ zip_subset.py
```

현재 핵심 공통 로직은 `traffic_ai_core.data.zip_subset`에 위치한다.

## 운영 원칙

- 공통 로직은 가능한 한 이 모듈에 모은다.
- 실행 앱별 중복 구현은 지양한다.
- 앱 전용 책임은 `apps/`에 두고, 공통 책임은 `packages/ai-core`에 둔다.
- 특정 개인 환경에 종속적인 경로 하드코딩은 코어 모듈에 넣지 않는다.

## 참고

- 이 모듈은 단독 실행용 프로그램이 아니다.
- 실제 실행 명령은 `apps/training-runner`, `apps/inference-service`에서 관리한다.
- Python 설치 및 실행 순서는 `docs/python_setup.md`를 함께 참고한다.
- subset 생성 방식과 데이터셋 처리 기준은 `docs/dataset_notes.md`를 따른다.