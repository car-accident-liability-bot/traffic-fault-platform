# backend-api-java

React 프론트엔드와 Python `inference-service` 사이를 연결하는 **별도 Java 백엔드 모듈**이다.

## 목적

- 프론트엔드 업로드 요청 수신 (`/api/analyze`)
- 프론트 질문 문자열을 Python 추론용 `question_type` 으로 변환
- `inference-service`의 `/predict` 호출
- 나중에 **이 모듈만 단독으로 빌드해서 JAR 배포** 가능하도록 분리

## 현재 반영된 구조

- 포트: `28080`
- Python inference 기본 주소: `http://localhost:8000`
- 추론 엔드포인트: `/predict`
- 프론트와의 계약:
  - `video`: multipart 파일
  - `question`: 프론트 질문 제목 문자열

## 질문 매핑

프론트 질문 제목 → Python `question_type`

- `해당 사고는 어떤 도로 환경에서 발생했는가?` → `accident_place`
- `이 사고 장소의 특징은 무엇인가?` → `accident_place_feature`
- `이 사고의 과실비율은 어떻게 되는가?` → `fault_ratio`
- `과실비율 기준으로 더 큰 과실을 가진 차량은 누구인가?` → `fault_compare`

## 실행 설정

`src/main/resources/application.yml`

```yaml
server:
  port: 28080

traffic:
  inference:
    base-url: ${TRAFFIC_INFERENCE_BASE_URL:http://localhost:8000}
    predict-path: ${TRAFFIC_INFERENCE_PREDICT_PATH:/predict}
```

## 빌드

이 모듈 디렉터리에서 실행:

```bash
cd apps/backend-api-java
gradle bootJar
```

생성 결과물:

```text
build/libs/backend-api-java.jar
```

> 참고: 현재 전달본에는 Gradle Wrapper(`gradlew`, `gradlew.bat`, `gradle/wrapper/*`)는 포함하지 않았다.
> 로컬에 Gradle이 설치되어 있다면 바로 `gradle bootJar`로 빌드하면 되고,
> 필요하면 이 모듈 디렉터리에서 한 번 `gradle wrapper`를 생성한 뒤 `./gradlew bootJar` 형태로 써도 된다.


## 실행

```bash
cd apps/backend-api-java
TRAFFIC_INFERENCE_BASE_URL=http://localhost:8000 gradle bootRun
```

또는 빌드 후:

```bash
java -jar build/libs/backend-api-java.jar
```

## API

### 1) 헬스체크

```text
GET /api/health
```

### 2) 질문 목록

```text
GET /api/questions
```

### 3) 분석 요청

```text
POST /api/analyze
Content-Type: multipart/form-data
```

폼 필드:

- `video`: 영상 파일
- `question`: 프론트 질문 제목

## 구현 메모

- inference 쪽 Python 소스는 수정하지 않는다.
- Java 백엔드는 업로드 영상을 임시 파일로 내렸다가 Python API로 다시 전달한다.
- `getBytes()` 대신 `transferTo()` 기반으로 처리해 대용량 업로드에서 힙 사용량을 줄였다.
