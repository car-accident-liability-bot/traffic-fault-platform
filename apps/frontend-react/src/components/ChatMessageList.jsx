function WelcomeCard({ selectedVideoName, selectedQuestionTitle }) {
  return (
    <div className="welcome-card">
      <span className="welcome-card-badge">Quick Start</span>
      <h2>영상 첨부 → 질문 선택 → 전송</h2>
      

      <div className="welcome-card-grid">
        <div className="welcome-card-item">
          <span>업로드 영상</span>
          <strong>{selectedVideoName || '아직 선택되지 않음'}</strong>
        </div>
        <div className="welcome-card-item">
          <span>선택 질문</span>
          <strong>{selectedQuestionTitle || '아직 선택되지 않음'}</strong>
        </div>
      </div>
    </div>
  );
}

export default function ChatMessageList({ messages, selectedVideoName, selectedQuestionTitle }) {
  return (
    <section className="chat-message-list">
      {messages.length === 1 ? (
        <WelcomeCard
          selectedVideoName={selectedVideoName}
          selectedQuestionTitle={selectedQuestionTitle}
        />
      ) : null}

      {messages.map((message) => {
        const isUser = message.role === 'user';

        return (
          <div
            key={message.id}
            className={`chat-message-row ${isUser ? 'is-user' : 'is-assistant'}`}
          >
            <article className={`chat-message-card ${isUser ? 'user' : 'assistant'}`}>
              <div className="chat-message-top">
                <span className={`chat-message-badge ${isUser ? 'user' : 'assistant'}`}>
                  {message.badge || (isUser ? 'User Input' : 'Assistant Reply')}
                </span>
              </div>

              <p className="chat-message-text">{message.text}</p>

              {message.meta ? <span className="chat-message-meta">{message.meta}</span> : null}
            </article>
          </div>
        );
      })}
    </section>
  );
}
