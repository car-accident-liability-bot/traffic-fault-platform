# traffic-training-runner

데이터셋 subset 생성, 학습 실행, 오프라인 검증용 러너 모듈이다.

## 이 모듈을 쓰기 전에 먼저 해야 할 것

이 모듈은 `traffic_ai_core`에 의존한다.  
따라서 저장소 루트에서 아래 순서로 설치해야 한다.

```bash
python -m pip install -e packages/ai-core
python -m pip install -e apps/training-runner
```

## 기본 실행 방식

이 저장소의 subset 생성은 **공용 URL 기준 실행**을 기본으로 한다.  
전체 ZIP을 로컬로 복사하지 않고, 공개 디렉터리 인덱스와 HTTP Range 읽기를 사용해서 필요한 ZIP 정보만 읽는다.

```bash
python -m training_runner.cli.build_pipeline_subset \
  --raw-url "https://data.taeo-dev.com/dataset/traffic/1.Training/raw_data_231108_add/" \
  --label-url "https://data.taeo-dev.com/dataset/traffic/1.Training/label_data_231108_add/"
```

## 기본 동작 및 출력물

기본 동작은 아래와 같다.

- raw/label ZIP은 ZIP 내부 basename overlap 기준으로 1:1 매칭한다.
- 라벨 JSON에서 과실비율 누락/`NaN`/placeholder 값, 도로유형 미상·불명·기타, case code 누락/비정상, 희귀 case code, 핵심 필드가 비어 있는 JSON은 전처리 단계에서 제외한다.
- manifest와 summary는 저장소 루트의 `data/manifests/` 아래에 저장한다.
- working subset 추출은 저장소 루트의 `data/working/pipeline_subset/` 아래에 저장한다.
- manifest에는 로컬 절대경로를 저장하지 않는다.

기본적으로 아래 출력물을 생성한다.

```text
data/manifests/subset_pipeline.csv
data/manifests/subset_pipeline.json
data/manifests/subset_pipeline_summary.json
data/working/pipeline_subset/
```

## 로컬 디렉터리 fallback 실행

운영 환경에서 공용 URL 대신 로컬 NAS 마운트 경로를 직접 읽고 싶다면 아래처럼 실행할 수 있다.

```bash
python -m training_runner.cli.build_pipeline_subset \
  --raw-dir "/path/to/traffic/1.Training/raw_data_231108_add" \
  --label-dir "/path/to/traffic/1.Training/label_data_231108_add"
```

## 전처리 옵션

```bash
python -m training_runner.cli.build_pipeline_subset \
  --raw-url "https://data.taeo-dev.com/dataset/traffic/1.Training/raw_data_231108_add/" \
  --label-url "https://data.taeo-dev.com/dataset/traffic/1.Training/label_data_231108_add/" \
  --rare-case-code-min-frequency 2
```

추가 옵션:

- `--disable-label-quality-filter`: 라벨 품질 전처리를 끈다.
- `--rare-case-code-min-frequency 1`: 희귀 case code 제외를 끈다.
- `--allow-missing-case-code`: 기본 전처리의 case code 누락/비정상 제외를 끄고 유지한다.

## URL 모드 주의사항

- 입력 URL은 디렉터리 인덱스가 열리는 주소여야 한다.
- 원격 ZIP 직접 읽기를 위해 서버가 HTTP byte-range 요청을 허용해야 한다.
- 실행 결과 생성되는 manifest, summary, working subset은 공용 NAS에 저장되지 않고, 명령을 실행한 사용자의 로컬 저장소 `<repo-root>/data/...` 아래에 생성된다.