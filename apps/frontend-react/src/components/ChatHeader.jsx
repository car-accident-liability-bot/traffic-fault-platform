export default function ChatHeader() {
  return (
    <header className="chat-header-shell">
      <div>
        <div className="chat-header-badge">Traffic Accident Analysis</div>
        <h1 className="chat-header-title">차대차 사고 분석 챗봇</h1>
        <p className="chat-header-description">
          사고 영상을 첨부하고 질문을 선택하면 과실비율, 도로 환경, 장소 특징을 한 화면에서 바로 확인할 수 있습니다.
        </p>
      </div>

      <div className="chat-header-status-card">
        <span className="chat-header-status-dot" />
        <div>
          <strong>Prototype Ready</strong>
          <p>video + question 멀티파트 전송 구조</p>
        </div>
      </div>
    </header>
  );
}
