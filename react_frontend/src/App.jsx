import LearningPage from "./learning/pages/LearningPage.jsx";
import HomePage from "./learning/pages/HomePage.jsx";
import "./learning/learning.css";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";
import { apiRequest } from "./api";
import { useLearningStats } from "./useLearningStats";
import LearningDashboard from "./LearningDashboard";
import WeaknessTraining from "./WeaknessTraining";
import SummaryNotes from "./SummaryNotes";
import MockExam from "./MockExam";
function App() {
  // =========================================================
  // 공통 상태
  // =========================================================
  const [currentPage, setCurrentPage] = useState("home");
  // 개념학습실 / 문제풀이실은 프로필 진입 여부와 별개로 유지합니다.
  const [activeArea, setActiveArea] = useState("study");
  const [profileReturnPage, setProfileReturnPage] = useState("learn");
  const [darkMode, setDarkMode] = useState(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const [selectedCert, setSelectedCert] = useState("정보처리기사");
  const [certifications, setCertifications] = useState([]);
  const [connectionError, setConnectionError] = useState("");
  const [certReload, setCertReload] = useState(0);
  const [conversationId, setConversationId] = useState(null);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");
  const [quizSubmitting, setQuizSubmitting] = useState(false);
  const chatRequest = useRef(null);
  const quizRequest = useRef(null);
  const submitRequest = useRef(null);
  const selectedCertId = certifications.find(cert => cert.label === selectedCert)?.id;

  useEffect(() => {
    const controller = new AbortController();
    apiRequest("/api/certifications", { signal: controller.signal })
      .then(data => {
        setCertifications(data.certifications);
        setConnectionError("");
      })
      .catch(error => {
        if (error.name !== "AbortError") setConnectionError(error.message);
      });
    return () => controller.abort();
  }, [certReload]);

  useEffect(() => () => {
    chatRequest.current?.abort();
    quizRequest.current?.abort();
    submitRequest.current?.abort();
  }, []);

  // =========================================================
  // 대시보드와 취약점 분석은 동일한 서버 통계를 사용합니다.
  const { stats, loading: statsLoading, error: statsError, refresh: refreshStats } = useLearningStats(selectedCertId, currentPage);
  const refreshLearningData = () => {
    if (!selectedCertId) setCertReload(value => value + 1);
    refreshStats();
  };

  // =========================================================
  // 취약점 개념 요약은 SummaryNotes에서 서버 데이터로 표시합니다.
  // =========================================================

  // =========================================================
  // 모의고사는 MockExam에서 서버 문제은행과 연결합니다.
  // =========================================================
  // AI 채팅 상태
  // =========================================================
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [answerLength, setAnswerLength] = useState("medium");
  // =========================================================
  // 문제풀이 상태
  // =========================================================
  const [quiz, setQuiz] = useState(null);
  const [quizLoading, setQuizLoading] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [quizResult, setQuizResult] = useState(null);
  const [quizError, setQuizError] = useState("");
  const [lessonPractice, setLessonPractice] = useState(null);
  const [learningReturnId, setLearningReturnId] = useState(null);

  // 문제풀이 소크라테스 문답 상태
  // 현재는 프론트엔드 시연용이며, 추후 AI/DB 연동 시 실제 이해도 판정으로 교체할 수 있습니다.
  const [socraticOpen, setSocraticOpen] = useState(false);
  const [socraticAnswer, setSocraticAnswer] = useState("");
  const [socraticMessages, setSocraticMessages] = useState([]);
  const [socraticCompleted, setSocraticCompleted] = useState(false);
  // =========================================================
  // =========================================================
  // 취약점 학습 상태
  // =========================================================
  // =========================================================
  // 모의고사 상태
  // =========================================================
  // AI 채팅
  // =========================================================
  const sendMessage = async () => {
    const text = input.trim();
    if (!text || chatRequest.current || !selectedCertId) return;
    const controller = new AbortController();
    chatRequest.current = controller;
    setChatLoading(true);
    setChatError("");
    setMessages(prev => [...prev, { id: `${Date.now()}-user`, role: "user", content: text }]);
    setInput("");
    try {
      const data = await apiRequest("/api/chat", {
        signal: controller.signal,
        body: { message: text, cert: selectedCertId, answer_length: answerLength,
                conversation_id: conversationId },
      });
      if (chatRequest.current !== controller) return;
      setConversationId(data.conversation_id);
      setMessages(prev => [...prev, { id: `${Date.now()}-assistant`, role: "assistant", content: data.answer }]);
    } catch (error) {
      if (chatRequest.current !== controller || error.name === "AbortError") return;
      setChatError(error.message);
      setInput(text);
    } finally {
      if (chatRequest.current === controller) {
        chatRequest.current = null;
        setChatLoading(false);
      }
    }
  };
  const handleKeyDown = e => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      sendMessage();
    }
  };
  const resetChat = () => {
    chatRequest.current?.abort();
    chatRequest.current = null;
    setChatLoading(false);
    setConversationId(null);
    setChatError("");
    setMessages([]);
    setInput("");
  };
  const newChat = () => {
    resetChat();
    setActiveArea("practice");
    setCurrentPage("chat");
  };

  const resetSocraticPractice = () => {
    setSocraticOpen(false);
    setSocraticAnswer("");
    setSocraticMessages([]);
    setSocraticCompleted(false);
  };

  const submitSocraticAnswer = () => {
    const answer = socraticAnswer.trim();
    if (!answer || socraticCompleted) return;

    const previousUserTurns = socraticMessages.filter(message => message.role === "user").length;
    const userMessage = {
      id: `${Date.now()}-socratic-user`,
      role: "user",
      content: answer,
    };

    if (previousUserTurns === 0) {
      const followUp = answer.length < 35
        ? "좋아요. 한 단계 더 구체적으로 설명해볼까요? 정답 선택지가 왜 맞는지 핵심 용어나 원리를 포함해서 적어보세요."
        : "좋아요. 이제 오답 선택지 중 하나를 골라, 그 선택지가 왜 틀렸는지도 자신의 말로 설명해볼까요?";

      setSocraticMessages(prev => [
        ...prev,
        userMessage,
        {
          id: `${Date.now()}-socratic-assistant`,
          role: "assistant",
          content: followUp,
        },
      ]);
    } else {
      setSocraticMessages(prev => [
        ...prev,
        userMessage,
        {
          id: `${Date.now()}-socratic-complete`,
          role: "assistant",
          content: "좋습니다. 정답의 근거와 오답의 이유를 자신의 말로 설명해봤어요. 이렇게 설명할 수 있으면 단순 암기보다 개념을 더 오래 기억하는 데 도움이 됩니다.",
        },
      ]);
      setSocraticCompleted(true);
    }

    setSocraticAnswer("");
  };
  const changeCertification = label => {
    if (label === selectedCert) return;
    // 자격증 변경 시 대화 기록만 초기화하며 프로필/학습실 위치는 유지합니다.
    resetChat();
    quizRequest.current?.abort();
    submitRequest.current?.abort();
    quizRequest.current = null;
    submitRequest.current = null;
    setQuizLoading(false);
    setQuizSubmitting(false);
    setQuiz(null);
    setSelectedAnswer(null);
    setQuizResult(null);
    setQuizError("");
    resetSocraticPractice();
    setSelectedCert(label);
    setLessonPractice(null);
  };
  const loadQuiz = async (requestedLesson = null) => {
    const scope = requestedLesson?.id ? requestedLesson : null;
    setCurrentPage("quiz");
    resetSocraticPractice();
    if (quizRequest.current || submitRequest.current) return;
    setLessonPractice(scope);
    if (!scope && !selectedCertId) {
      setQuiz(null);
      setQuizError(connectionError
        ? `자격증 목록 API에 연결하지 못했습니다. ${connectionError}`
        : "자격증 목록을 불러온 뒤 다시 시도해 주세요. API 서버 실행 여부를 확인해 주세요.");
      return;
    }
    const controller = new AbortController();
    quizRequest.current = controller;
    setQuizLoading(true);
    setQuiz(null);
    setQuizResult(null);
    setSelectedAnswer(null);
    setQuizError("");
    try {
      const data = await apiRequest("/api/quiz", {
        signal: controller.signal, body: scope ? { cert: 'EIP', mode: 'lesson', lesson_id: scope.id } : { cert: selectedCertId },
      });
      if (quizRequest.current === controller) setQuiz(data);
    } catch (error) {
      if (quizRequest.current === controller && error.name !== "AbortError") setQuizError(error.message);
    } finally {
      if (quizRequest.current === controller) {
        quizRequest.current = null;
        setQuizLoading(false);
      }
    }
  };
  const submitQuiz = async () => {
    if (!quiz || selectedAnswer === null || quizResult || submitRequest.current) return;
    const controller = new AbortController();
    submitRequest.current = controller;
    setQuizSubmitting(true);
    setQuizError("");
    try {
      const data = await apiRequest("/api/quiz/submit", {
        signal: controller.signal,
        body: { attempt_id: quiz.attempt_id, selected_answer: selectedAnswer },
      });
      if (submitRequest.current === controller) {
        setQuizResult(data);
        refreshStats();
      }
    } catch (error) {
      if (submitRequest.current === controller && error.name !== "AbortError") setQuizError(error.message);
    } finally {
      if (submitRequest.current === controller) {
        submitRequest.current = null;
        setQuizSubmitting(false);
      }
    }
  };
  // =========================================================
  // 취약점 화면은 WeaknessTraining에서 서버 출제·채점을 처리합니다.
  const openWeaknessPage = () => setCurrentPage("weakness");
  // =========================================================
  // 취약점 개념 요약
  // =========================================================
  const openSummaryPage = () => setCurrentPage("summary");
  // =========================================================
  // 모의고사
  // =========================================================
  const openMockExamPage = () => setCurrentPage("mockexam");
  // currentPage === "profile"이어도 기존 학습실 메뉴를 그대로 유지합니다.
  const studyMode = activeArea === "study";
  // 문제풀이실로 명시적으로 이동할 때만 activeArea를 전환합니다.
  const openPractice = () => {
    setActiveArea("practice");
    void loadQuiz();
  };
  const openStudy = () => {
    setActiveArea("study");
    setCurrentPage("learn");
  };
  const openLessonPractice = lesson => {
    quizRequest.current?.abort();
    quizRequest.current = null;
    setSelectedCert("정보처리기사");
    setLearningReturnId(lesson.id);
    setActiveArea("practice");
    void loadQuiz(lesson);
  };
  const openStudyChat = () => {
    setActiveArea("study");
    setCurrentPage("studyChat");
  };
  const openProfile = () => {
    if (currentPage === "profile") return;
    setProfileReturnPage(currentPage);
    setCurrentPage("profile");
  };
  const returnFromProfile = () => setCurrentPage(profileReturnPage);
  // 헤더 제목을 누르면 홈 화면으로 이동
  const goHome = () => setCurrentPage("home");
  return <div className={`app ${darkMode ? "dark" : "light"} ${currentPage === "home" ? "home-layout" : ""}`}>
      {/* =====================================================
          Sidebar
       ===================================================== */}
      {currentPage !== "home" && <aside className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
        <div className="sidebar-top">
          <button type="button" className="learning-return-button" onClick={studyMode ? openPractice : openStudy} title={studyMode ? "문제풀이 화면으로 가기" : "개념 학습으로 돌아가기"}>
            <span className="menu-icon">{studyMode ? "□" : "‹"}</span>
            {!sidebarCollapsed && <span>{studyMode ? "문제풀이로 가기" : "개념 학습으로 돌아가기"}</span>}
          </button>
          <div className="sidebar-header">
            <button type="button" className="sidebar-toggle" onClick={() => setSidebarCollapsed(prev => !prev)} title={sidebarCollapsed ? "사이드바 펼치기" : "사이드바 접기"}>
              {sidebarCollapsed ? "›" : "‹"}
            </button>
          </div>
          {/* 메뉴 */}
          <nav className="menu">
            {studyMode ? <>
              <button type="button" className={`menu-item ${currentPage === "learn" ? "active" : ""}`} onClick={openStudy}>
                <span className="menu-icon">▤</span>{!sidebarCollapsed && <span>개념 학습</span>}
              </button>
              <button type="button" className={`menu-item ${currentPage === "studyChat" ? "active" : ""}`} onClick={openStudyChat}>
                <span className="menu-icon">◉</span>{!sidebarCollapsed && <span>AI 튜터</span>}
              </button>
            </> : <>
              <button type="button" className={`menu-item ${currentPage === "chat" ? "active" : ""}`} onClick={() => setCurrentPage("chat")}>
                <span className="menu-icon">◉</span>{!sidebarCollapsed && <span>AI 튜터</span>}
              </button>
              <button type="button" className={`menu-item ${currentPage === "quiz" ? "active" : ""}`} onClick={loadQuiz}>
                <span className="menu-icon">□</span>{!sidebarCollapsed && <span>문제 풀이</span>}
              </button>
              <button type="button" className={`menu-item ${currentPage === "weakness" ? "active" : ""}`} onClick={openWeaknessPage}>
                <span className="menu-icon">◇</span>{!sidebarCollapsed && <span>취약점 학습</span>}
              </button>
              <button type="button" className={`menu-item ${currentPage === "summary" ? "active" : ""}`} onClick={openSummaryPage}>
                <span className="menu-icon">○</span>{!sidebarCollapsed && <span>취약점 개념 요약</span>}
              </button>
              <button type="button" className={`menu-item ${currentPage === "mockexam" ? "active" : ""}`} onClick={openMockExamPage}>
                <span className="menu-icon">△</span>{!sidebarCollapsed && <span>모의고사</span>}
              </button>
            </>}
          </nav>
        </div>
        {/* 사용자 */}
        <div className="sidebar-bottom">
          <button type="button" className={`sidebar-user-button ${currentPage === "profile" ? "active" : ""}`} onClick={openProfile} title="나의 학습 현황">
            <div className="profile-circle">
              U
            </div>
            {!sidebarCollapsed && <div className="profile-info">
                <div className="profile-name">
                  서정우
                </div>
                <div className="profile-subtitle">
                  {selectedCert}
                </div>
              </div>}
          </button>
        </div>
      </aside>}
      {/* =====================================================
          Main
       ===================================================== */}
      <main className="main">
        {/* Header */}
        <header className="header">
          <button type="button" className="header-title header-home-button" onClick={goHome} title="홈으로">
            안양대학교 AI Tutor
          </button>
          <button type="button" className="theme-toggle" onClick={() => setDarkMode(prev => !prev)} title={darkMode ? "라이트 모드" : "다크 모드"}>
            {darkMode ? "☀" : "☾"}
          </button>
        </header>
        {currentPage === "home" && <div className="study-home"><HomePage onStart={openStudy} /></div>}
        {currentPage === "learn" && <div className="concept-learning"><LearningPage returnLessonId={learningReturnId} onLessonChange={setLearningReturnId} onPractice={openLessonPractice} /></div>}
        {/* =================================================
            AI Tutor
         ================================================= */}
        {(currentPage === "chat" || currentPage === "studyChat") && <>
            <section className="chat-area">
              {messages.length === 0 ? <div className="welcome">
                  <div className="welcome-logo">
                    AI
                  </div>
                  <h1>
                    무엇을 공부할까요?
                  </h1>
                  <p>
                    {selectedCert} 학습 중 궁금한 내용을
                    자유롭게 질문해보세요.
                  </p>
                  <div className="suggestions">
                    <button type="button" onClick={() => setInput("데이터베이스 정규화에 대해 설명해줘")}>
                      데이터베이스 정규화 설명
                    </button>
                    <button type="button" onClick={() => setInput("OSI 7계층을 쉽게 설명해줘")}>
                      OSI 7계층 설명
                    </button>
                    <button type="button" onClick={() => setInput("TCP와 UDP의 차이점을 알려줘")}>
                      TCP와 UDP 차이
                    </button>
                  </div>
                </div> : <div className="messages">
                  {messages.map(message => <div key={message.id} className={`message-row ${message.role}`}>
                        {message.role === "assistant" && <div className="assistant-avatar">
                            AI
                          </div>}
                        <div className="message">
                          {message.role === "assistant" ? <ReactMarkdown remarkPlugins={[remarkGfm]}>
                              {message.content}
                            </ReactMarkdown> : message.content}
                        </div>
                      </div>)}
                                            
                  {chatLoading && <div className="message-row assistant" role="status">
                      <div className="assistant-avatar">
                        AI
                      </div>
                      <div className="message chat-typing">
                        <span className="chat-typing-spinner" />
                        답변을 준비하고 있습니다.
                      </div>
                    </div>}
                
                </div>}
            </section>
            {/* 채팅 입력 */}
            <div className="input-section">
              {connectionError && <div role="alert">{connectionError} <button type="button" onClick={() => setCertReload(value => value + 1)}>연결 다시 확인</button></div>}
              {chatError && <div role="alert">{chatError}</div>}
            
                                          <div className="answer-length-control">
                <button
                  type="button"
                  className="chat-reset-button"
                  onClick={() => {
                    if (messages.length === 0 && !chatError) return;
                    if (window.confirm("대화 내용을 모두 지울까요?")) resetChat();
                  }}
                >
                  대화 초기화
                </button>
                <span className="answer-length-label">
                  답변 길이
                </span>
                <button type="button" className={answerLength === "short" ? "active" : ""} onClick={() => setAnswerLength("short")}>
                  간단히
                </button>
                <button type="button" className={answerLength === "medium" ? "active" : ""} onClick={() => setAnswerLength("medium")}>
                  보통
                </button>
                <button type="button" className={answerLength === "long" ? "active" : ""} onClick={() => setAnswerLength("long")}>
                  자세히
                </button>
              </div>
                
              <div className="input-container">
                <textarea disabled={chatLoading} maxLength={8000} value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKeyDown} placeholder="AI 튜터에게 질문해보세요" rows={1} />
                <button type="button" className="send-button" onClick={sendMessage} disabled={!input.trim() || chatLoading || !selectedCertId}>
                  ↑
                </button>
              </div>
              <div className="input-caption">
                <p>
                  © 2026 자격증 변형 문제 출제 AI 튜터 시스템
                  (AI Tutor Project Team). All Rights Reserved.
                </p>
                <p>
                  본 시스템은 캡스톤 디자인 졸업과제용으로
                  제작되었으며, 무단 복제 및 전재를 금합니다.
                </p>
                <p>
                  자료 출처 : 시나공 기출문제집 정보처리기사 필기 /
                  국가법령정보센터
                </p>
              </div>
            </div>
          </>}
        {/* =================================================
            문제 풀이
         ================================================= */}
        {currentPage === "quiz" && <section className="quiz-page">
            <div className="quiz-header">
              <div>
                <span className="quiz-page-label">PRACTICE</span>
                <h1>문제 풀이</h1>
                <p>{lessonPractice ? `${lessonPractice.title} · 학습한 단원의 자료를 바탕으로 만든 문제입니다.` : 'AI가 기출 데이터를 분석해 만든 변형 문제입니다.'}</p>
                {lessonPractice && <button type="button" className="quiz-new-button" onClick={openStudy}>학습하던 단원으로 돌아가기</button>}
              </div>
              <button type="button" className="quiz-new-button" onClick={() => loadQuiz(lessonPractice)} disabled={quizLoading || quizSubmitting || (!lessonPractice && !selectedCertId)}>
                새 문제
              </button>
            </div>
            {quizLoading && <div className="quiz-loading">
                <div className="quiz-loading-spinner" />
                <h3>AI가 문제를 생성하고 검토하고 있습니다.</h3>
                <p>출제 후 검수까지 진행하므로 잠시 시간이 걸릴 수 있습니다.</p>
              </div>}
            {!quizLoading && quizError && <div className="quiz-error">{quizError}</div>}
            {!quizLoading && quiz && <div className={`quiz-workspace ${quizResult ? "has-result" : ""} ${socraticOpen ? "socratic-active" : ""}`}>
                {/* 왼쪽: 원래 문제 */}
                <div className="quiz-card quiz-question-panel">
                  <div className="quiz-meta">
                    <span>{lessonPractice ? '정보처리기사' : selectedCert}</span>
                    <span>{quiz.topic_label}</span>
                  </div>
                  <h2 className="quiz-question">
                    Q. {quiz.question}
                  </h2>
                  {quiz.code_block && <pre className="quiz-code">
                      <code>{quiz.code_block}</code>
                    </pre>}
                  {quiz.table_data && <div className="quiz-table">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {quiz.table_data}
                      </ReactMarkdown>
                    </div>}
                  <div className="quiz-options">
                    {quiz.options.map((option, index) => {
                const number = index + 1;
                let optionClass = "quiz-option";
                if (selectedAnswer === number) {
                  optionClass += " selected";
                }
                if (quizResult) {
                  if (number === quizResult.correct_answer) {
                    optionClass += " correct";
                  } else if (number === selectedAnswer && !quizResult.is_correct) {
                    optionClass += " incorrect";
                  }
                }
                return <button type="button" key={number} className={optionClass} disabled={!!quizResult || quizSubmitting} onClick={() => setSelectedAnswer(number)}>
                          <span className="quiz-option-number">{number}</span>
                          <span>
                            {option.replace(/^\s*\d+\s*[).:-]?\s*/, "")}
                          </span>
                        </button>;
              })}
                  </div>
                  {!quizResult && <div className="quiz-submit-area">
                      <button type="button" className="quiz-submit-button" disabled={selectedAnswer === null || quizSubmitting} onClick={submitQuiz}>
                        {quizSubmitting ? "채점 중…" : "정답 제출"}
                      </button>
                    </div>}
                  {quizResult && <div className="quiz-inline-result">
                      <div className={`quiz-result-title ${quizResult.is_correct ? "success" : "wrong"}`}>
                        {quizResult.is_correct ? "정답입니다!" : "오답입니다."}
                      </div>
                      {!quizResult.is_correct && <p className="quiz-correct-answer">
                          정답은 <strong>{quizResult.correct_answer}번</strong>입니다.
                        </p>}
                    </div>}
                </div>
                {/* 오른쪽: AI 해설 / 소크라테스 문답을 같은 자리에서 전환 */}
                {quizResult && <aside className="quiz-solution-panel">
                    <div className="quiz-solution-header">
                      <span className="quiz-solution-label">
                        {socraticOpen ? "SOCRATIC REVIEW" : "AI SOLUTION"}
                      </span>
                      <h2>{socraticOpen ? "서술해보기" : "AI 해설"}</h2>
                      <p>
                        {socraticOpen
                          ? "정답을 외우는 대신, 문제의 핵심 개념을 자신의 말로 설명해보세요."
                          : "문제와 비교하면서 해설을 확인해보세요."}
                      </p>
                    </div>

                    {!socraticOpen ? (
                      <div className="quiz-explanation quiz-explanation-side">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {quizResult.explanation}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <div className="quiz-socratic-panel">
                        <div className="quiz-socratic-message assistant">
                          <span className="quiz-socratic-avatar">AI</span>
                          <div>
                            <strong>먼저 이렇게 설명해볼까요?</strong>
                            <p>
                              “{quiz.topic_label}”에서 이 문제가 묻는 핵심 개념은 무엇이고,
                              왜 {quizResult.correct_answer}번이 정답인지 자신의 말로 설명해보세요.
                            </p>
                          </div>
                        </div>

                        {socraticMessages.map(message => (
                          <div
                            key={message.id}
                            className={`quiz-socratic-message ${message.role}`}
                          >
                            {message.role === "assistant" && (
                              <span className="quiz-socratic-avatar">AI</span>
                            )}
                            <div>
                              {message.role === "user" && <strong>나의 설명</strong>}
                              <p>{message.content}</p>
                            </div>
                          </div>
                        ))}

                        {!socraticCompleted && (
                          <div className="quiz-socratic-input-wrap">
                            <textarea
                              value={socraticAnswer}
                              onChange={event => setSocraticAnswer(event.target.value)}
                              placeholder="내가 이해한 내용을 서술해보세요..."
                              rows={3}
                            />
                            <button
                              type="button"
                              onClick={submitSocraticAnswer}
                              disabled={!socraticAnswer.trim()}
                            >
                              설명 제출
                            </button>
                          </div>
                        )}

                        {socraticCompleted && (
                          <div className="quiz-socratic-complete">
                            <strong>설명 학습 완료</strong>
                            <span>
                              현재는 프론트엔드 시연용입니다. 실제 이해도 판정은 추후
                              AI/DB 연동 시 연결할 수 있습니다.
                            </span>
                          </div>
                        )}

                        {socraticMessages.length > 0 && (
                          <button
                            type="button"
                            className="quiz-socratic-reset"
                            onClick={() => {
                              setSocraticAnswer("");
                              setSocraticMessages([]);
                              setSocraticCompleted(false);
                            }}
                          >
                            처음부터 다시 설명하기
                          </button>
                        )}
                      </div>
                    )}

                    <div className="quiz-solution-actions">
                      <button
                        type="button"
                        className={`quiz-socratic-toggle ${socraticOpen ? "active" : ""}`}
                        onClick={() => setSocraticOpen(prev => !prev)}
                      >
                        {socraticOpen ? "💡 해설보기" : "✎ 서술해보기"}
                      </button>
                      <button
                        type="button"
                        className="quiz-next-button"
                        onClick={() => loadQuiz(lessonPractice)}
                      >
                        다음 문제
                      </button>
                    </div>
                  </aside>}
              </div>}
          </section>}
        {/* =================================================
            취약점 집중 학습
         ================================================= */}
        {currentPage === "weakness" && <WeaknessTraining
          key={selectedCert}
          cert={selectedCertId} label={selectedCert} stats={stats}
          loading={statsLoading || (!selectedCertId && !connectionError)} error={statsError || connectionError}
          onRefresh={refreshLearningData} onGeneralPractice={loadQuiz}
        />}
        {/* =================================================
            취약점 개념 요약
         ================================================= */}
        {currentPage === "summary" && <SummaryNotes key={selectedCert} cert={selectedCertId} label={selectedCert}
          connectionError={connectionError} onReconnect={() => setCertReload(value => value + 1)} />}
        {/* =================================================
            모의고사
         ================================================= */}
        {currentPage === "mockexam" && <MockExam key={selectedCertId || selectedCert} cert={selectedCertId} label={selectedCert} onSubmitted={refreshStats} onWeakness={() => setCurrentPage("weakness")} />}
        {currentPage === "profile" && <section className="profile-page">
            <div className="profile-content">
              {/* 페이지 제목 */}
              <div className="profile-page-header">
                <div className="profile-page-heading">
                  <span className="profile-page-label">
                    LEARNING PROFILE
                  </span>
                  <h1>나의 학습 현황</h1>
                  <p>
                    학습 기록과 현재 학습 설정을 한눈에 확인할 수 있습니다.
                  </p>
                </div>
                <div className="profile-header-actions">
                  <button type="button" className="profile-back-button" onClick={returnFromProfile}>
                    ← 이전 화면
                  </button>
                  <button type="button" className="learning-reset-button" onClick={() => alert("학습 데이터 초기화 기능은 추후 연결 예정입니다.")}>
                    학습 데이터 초기화
                  </button>
                </div>
              </div>
              {/* =============================================
                  취약점 분석
               ============================================= */}
              <LearningDashboard stats={stats}
                loading={statsLoading || (!selectedCertId && !connectionError)} error={statsError || connectionError}
                onRetry={refreshLearningData} darkMode={darkMode} />
              <section className="cert-settings-section">
                <div className="cert-settings-header">
                  <span className="dashboard-small-label">
                    CERTIFICATE
                  </span>
                  <h2>
                    자격증 설정
                  </h2>
                  <p>
                    학습할 자격증을 선택하세요.
                  </p>
                </div>
                <div className="cert-grid">
                  {/* 정보처리기사 */}
                  <button type="button" className={`cert-card ${selectedCert === "정보처리기사" ? "selected" : ""}`} onClick={() => changeCertification("정보처리기사")}>
                    <div className="cert-card-top">
                      <div className="cert-icon">
                        IT
                      </div>
                      {selectedCert === "정보처리기사" && <div className="cert-check">
                          ✓
                        </div>}
                    </div>
                    <div className="cert-card-content">
                      <h3>
                        정보처리기사
                      </h3>
                      <p>
                        소프트웨어 설계 · 데이터베이스 ·
                        프로그래밍 언어 활용 ·
                        정보시스템 구축 관리
                      </p>
                    </div>
                    <div className="cert-card-footer">
                      국가기술자격
                    </div>
                  </button>
                  {/* 공인중개사 1차 */}
                  <button type="button" className={`cert-card ${selectedCert === "공인중개사 1차" ? "selected" : ""}`} onClick={() => changeCertification("공인중개사 1차")}>
                    <div className="cert-card-top">
                      <div className="cert-icon">
                        RE
                      </div>
                      {selectedCert === "공인중개사 1차" && <div className="cert-check">
                          ✓
                        </div>}
                    </div>
                    <div className="cert-card-content">
                      <h3>
                        공인중개사 1차
                      </h3>
                      <p>
                        부동산학개론 · 민법 및
                        민사특별법
                      </p>
                    </div>
                    <div className="cert-card-footer">
                      국가전문자격
                    </div>
                  </button>
                </div>
                <div className="current-cert-box">
                  <div>
                    <span>
                      현재 학습 자격증
                    </span>
                    <strong>
                      {selectedCert}
                    </strong>
                  </div>
                  <div className="current-cert-status">
                    선택됨
                  </div>
                </div>
              </section>
            </div>
          </section>}
      </main>
    </div>;
}
export default App;
