# traffic-inference-service

멀티모달 추론 서비스용 Python 앱 자리다.

현재는 팀 공용 구조 검증을 위한 최소 FastAPI 골격만 포함한다.

## 설치

저장소 루트에서 아래 순서로 설치한다.

```bash
python -m pip install -e packages/ai-core
python -m pip install -e apps/inference-service
```

## 실행

```bash
traffic-inference-service
```

또는

```bash
python -m traffic_inference_service.main
```

두 실행 방식 모두 동일하게 uvicorn 서버를 띄운다.

## health check

브라우저 또는 curl:

```text
GET http://localhost:8000/health
```
