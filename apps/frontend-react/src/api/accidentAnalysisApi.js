/**
 * 차대차 사고 분석 API 호출 전용 모듈
 *
 * 백엔드 기준 파라미터명:
 * - video: MultipartFile
 * - question: String
 *
 * 수정 포인트:
 * - 프론트는 상대 경로(/api)를 유지한다.
 * - 실제 포트 28080 연결은 Vite proxy / Nginx proxy 설정에서 처리한다.
 */
const API_BASE_PATH = '/api';
const ANALYZE_API_URL = `${API_BASE_PATH}/analyze`;

async function parseResponseBody(response) {
  const contentType = response.headers.get('content-type') || '';

  if (contentType.includes('application/json')) {
    return response.json();
  }

  const text = await response.text();
  return { message: text };
}

function normalizeError(error) {
  if (error instanceof Error) {
    return error;
  }

  return new Error('알 수 없는 오류가 발생했습니다.');
}

export async function analyzeAccidentVideo({ video, question }) {
  if (!(video instanceof File)) {
    throw new Error('유효한 영상 파일이 필요합니다.');
  }

  if (!video.type.startsWith('video/')) {
    throw new Error('영상 파일만 업로드할 수 있습니다.');
  }

  if (typeof question !== 'string' || question.trim() === '') {
    throw new Error('질문이 선택되지 않았습니다.');
  }

  const formData = new FormData();
  formData.append('video', video);
  formData.append('question', question.trim());

  try {
    const response = await fetch(ANALYZE_API_URL, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}`;

      try {
        const errorBody = await parseResponseBody(response);

        if (typeof errorBody?.message === 'string' && errorBody.message.trim() !== '') {
          errorMessage = errorBody.message;
        } else if (typeof errorBody?.error === 'string' && errorBody.error.trim() !== '') {
          errorMessage = errorBody.error;
        } else if (typeof errorBody?.detail === 'string' && errorBody.detail.trim() !== '') {
          errorMessage = errorBody.detail;
        }
      } catch {
        // 수정 포인트:
        // 에러 본문 파싱이 실패해도 HTTP 상태값은 유지한다.
      }

      throw new Error(errorMessage);
    }

    return await parseResponseBody(response);
  } catch (error) {
    throw normalizeError(error);
  }
}

export default analyzeAccidentVideo;
