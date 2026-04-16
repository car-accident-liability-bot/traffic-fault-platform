import { QUESTION_OPTIONS } from '../constants/questionOptions';

function formatFileSize(size) {
  if (!Number.isFinite(size) || size <= 0) return '0 B';

  const units = ['B', 'KB', 'MB', 'GB'];
  let value = size;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[unitIndex]}`;
}

export default function ChatComposer({
  selectedVideo,
  selectedQuestion,
  isSubmitting,
  canSend,
  onVideoChange,
  onRemoveVideo,
  onQuestionChange,
  onSend,
  fileInputRef,
}) {
  return (
    <div className="chat-composer-shell">
      {selectedVideo ? (
        <div className="attachment-strip">
          <div className="attachment-pill">
            <span className="attachment-pill-icon">VID</span>
            <div className="attachment-pill-text">
              <strong>{selectedVideo.name}</strong>
              <span>{formatFileSize(selectedVideo.size)}</span>
            </div>
          </div>

          <button
            type="button"
            className="attachment-remove-button"
            onClick={onRemoveVideo}
            disabled={isSubmitting}
          >
            제거
          </button>
        </div>
      ) : null}

      <div className="chat-composer-box">
        <label className="upload-trigger-button" title="사고 영상 첨부">
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={onVideoChange}
            disabled={isSubmitting}
            hidden
          />
          <span className="upload-trigger-icon">＋</span>
          <span className="upload-trigger-text">
            {selectedVideo ? '영상 변경' : '영상 첨부'}
          </span>
        </label>

        <div className="composer-divider" />

        <div className="composer-select-block">
          <label htmlFor="question-select" className="composer-hidden-label">
            질문 선택
          </label>
          <select
            id="question-select"
            className="composer-select"
            value={selectedQuestion}
            onChange={(event) => onQuestionChange(event.target.value)}
            disabled={isSubmitting}
          >
            <option value="">질문을 선택하세요</option>
            {QUESTION_OPTIONS.map((question) => (
              <option key={question.id} value={question.title}>
                {question.title}
              </option>
            ))}
          </select>
        </div>

        <button
          type="button"
          className="composer-send-button"
          disabled={!canSend}
          onClick={onSend}
        >
          {isSubmitting ? '전송 중...' : '보내기'}
        </button>
      </div>

      <div className="question-chip-row">
        {QUESTION_OPTIONS.map((question) => (
          <button
            key={question.id}
            type="button"
            className={`question-chip ${selectedQuestion === question.title ? 'is-active' : ''}`}
            disabled={isSubmitting}
            onClick={() => onQuestionChange(question.title)}
          >
            {question.shortLabel}
          </button>
        ))}
      </div>
    </div>
  );
}
