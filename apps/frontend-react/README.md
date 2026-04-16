# frontend-react

사고 영상 업로드와 질문 선택 UI를 제공하는 React 프론트엔드다.

## 현재 연동 대상

- 상대 경로 API: `/api/*`
- 개발 서버 프록시: `http://localhost:28080`
- 운영 Nginx 프록시: `http://backend-api-java:28080`

즉, 프론트 소스는 `/api/analyze`만 호출하고,
실제 포트 28080 라우팅은 프록시 설정에서 처리한다.
