import { useMemo, useRef, useState } from 'react';
import { analyzeAccidentVideo } from '../api/accidentAnalysisApi';
import ChatHeader from '../components/ChatHeader';
import ChatMessageList from '../components/ChatMessageList';
import ChatComposer from '../components/ChatComposer';
import { QUESTION_OPTIONS } from '../constants/questionOptions';
import '../styles/chatbot.css';

function extractAssistantText(payload) {
  if (!payload || typeof payload !== 'object') {
    return '응답 형식이 올바르지 않습니다.';
  }

  return (
    payload.answer ||
    payload.message ||
    payload.result ||
    '분석 결과가 도착했지만 표시할 응답 필드가 없습니다.'
  );
}

export default function AccidentChatPage() {
  const [selectedQuestion, setSelectedQuestion] = useState('');
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      badge: 'System Ready',
      text: '사고 영상을 첨부하고 질문을 선택하면 분석 요청을 바로 보낼 수 있습니다.'
    },
  ]);

  const fileInputRef = useRef(null);

  const canSend = useMemo(() => {
    return Boolean(selectedVideo) && Boolean(selectedQuestion) && !isSubmitting;
  }, [selectedVideo, selectedQuestion, isSubmitting]);

  const selectedQuestionItem = useMemo(() => {
    return QUESTION_OPTIONS.find((item) => item.title === selectedQuestion) || null;
  }, [selectedQuestion]);

  const handleVideoChange = (event) => {
    const file = event.target.files?.[0] ?? null;

    if (file && !file.type.startsWith('video/')) {
      window.alert('영상 파일만 업로드할 수 있습니다.');
      event.target.value = '';
      setSelectedVideo(null);
      return;
    }

    setSelectedVideo(file);
  };

  const handleRemoveVideo = () => {
    if (isSubmitting) return;

    setSelectedVideo(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSend = async () => {
    if (!selectedVideo || !selectedQuestion || isSubmitting) {
      return;
    }

    const userMessage = {
      id: Date.now(),
      role: 'user',
      badge: selectedQuestionItem?.shortLabel || '질문',
      text: selectedQuestion,
      meta: `첨부 영상: ${selectedVideo.name}`,
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsSubmitting(true);

    try {
      const payload = await analyzeAccidentVideo({
        video: selectedVideo,
        question: selectedQuestion,
      });

      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        badge: '분석 결과',
        text: extractAssistantText(payload),
        meta: 'API 응답 수신 완료',
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setSelectedQuestion('');
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.';

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 2,
          role: 'assistant',
          badge: '요청 실패',
          text: '분석 요청 중 오류가 발생했습니다.',
          meta: errorMessage,
        },
      ]);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="chatbot-page">
      <div className="chatbot-shell">
        <ChatHeader />

        <main className="chatbot-stage">
          <ChatMessageList
            messages={messages}
            selectedVideoName={selectedVideo?.name}
            selectedQuestionTitle={selectedQuestionItem?.title}
          />

          <ChatComposer
            selectedVideo={selectedVideo}
            selectedQuestion={selectedQuestion}
            isSubmitting={isSubmitting}
            canSend={canSend}
            onVideoChange={handleVideoChange}
            onRemoveVideo={handleRemoveVideo}
            onQuestionChange={setSelectedQuestion}
            onSend={handleSend}
            fileInputRef={fileInputRef}
          />
        </main>
      </div>
    </div>
  );
}
