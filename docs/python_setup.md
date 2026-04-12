# Python Setup Guide

이 문서는 프로젝트의 Python 실행 환경 준비, 패키지 설치, 기본 실행 흐름을 정리한 가이드다.

## 권장 Python 버전

- Python 3.11 이상

팀 공통 기준은 Python 3.11 이상으로 맞추는 것을 권장한다.  
현재 팀원이 대부분 Python 3.11 또는 3.12를 사용하고 있다면, 3.10 사용자는 가능하면 3.11 이상으로 맞추는 것을 권장한다.

## 1. 저장소 루트로 이동

예시:

```bash
cd traffic-fault-platform
```

## 2. 가상환경 생성 및 활성화

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. pip 업데이트

```bash
python -m pip install --upgrade pip
```

## 4. pyproject.toml 기반 패키지 설치

이 저장소는 Python 패키지를 각 모듈별 `pyproject.toml`로 관리한다.  
그래서 import 오류 없이 실행하려면 아래 3개를 설치해야 한다.

```bash
python -m pip install -e packages/ai-core
python -m pip install -e apps/training-runner
python -m pip install -e apps/inference-service
```

### 왜 `-e`를 쓰는가

`-e`는 editable install 이다.  
이렇게 설치하면 소스를 수정한 뒤 매번 재설치하지 않아도 현재 작업 중인 코드가 바로 반영된다.

## 5. 설치 확인

```bash
python -c "import traffic_ai_core; import training_runner; import traffic_inference_service; print('imports ok')"
```

정상이라면 아래처럼 출력된다.

```text
imports ok
```

## 6. subset 생성 실행 예시

기본 subset 생성은 공용 URL 기준으로 실행한다.

```bash
python -m training_runner.cli.build_pipeline_subset \
  --raw-url "https://data.taeo-dev.com/dataset/traffic/1.Training/raw_data_231108_add/" \
  --label-url "https://data.taeo-dev.com/dataset/traffic/1.Training/label_data_231108_add/"
```

공용 URL 대신 로컬 디렉터리를 직접 읽고 싶다면 아래처럼 실행할 수 있다.

```bash
python -m training_runner.cli.build_pipeline_subset \
  --raw-dir "/path/to/traffic/1.Training/raw_data_231108_add" \
  --label-dir "/path/to/traffic/1.Training/label_data_231108_add"
```

기본 출력 위치는 현재 작업 디렉터리가 아니라 저장소 루트 기준이다.

```text
data/manifests/subset_pipeline.csv
data/manifests/subset_pipeline.json
data/manifests/subset_pipeline_summary.json
data/working/pipeline_subset/
```

즉, 원본 데이터는 공용 URL 또는 로컬 입력 경로에서 읽고,  
생성되는 manifest, summary, working subset은 명령을 실행한 사용자의 로컬 저장소 `data/...` 아래에 생성된다.

기본 전처리는 아래를 포함한다.

- 과실비율 누락, `NaN`, 공백, `-`, `null`, `None` 등 placeholder 값인 JSON 제외
- 도로유형이 미상, 불명, 기타 등으로 해석되는 JSON 제외
- case code가 누락되었거나 비정상인 JSON 제외
- 핵심 라벨 필드가 사실상 비어 있는 JSON 제외
- 희귀 case code 제외

subset 생성의 상세 옵션은 `apps/training-runner/README.md`를 따른다.

## 7. inference-service 실행 예시

```bash
traffic-inference-service
```

또는

```bash
python -m traffic_inference_service.main
```

## 8. 트러블슈팅

### URL 모드에서 ZIP을 못 읽는 경우

아래를 먼저 확인한다.
- 입력 URL이 실제 디렉터리 인덱스 URL인지
- ZIP 파일 목록이 브라우저에서 열리는지
- 서버가 HTTP byte-range 요청을 허용하는지

### PowerShell에서 스크립트 실행이 막히는 경우

관리자 권한 PowerShell에서 아래를 한 번 실행한다.

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### `python` 대신 `py`만 되는 경우

Windows 환경에 따라 아래처럼 실행해도 된다.

```powershell
py -m venv .venv
py -m pip install --upgrade pip
py -m pip install -e packages/ai-core
py -m pip install -e apps/training-runner
py -m pip install -e apps/inference-service
```
