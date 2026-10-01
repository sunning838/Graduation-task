import { useState } from "react";

import ReactMarkdown from "react-markdown";

import remarkGfm from "remark-gfm";



import {

  RadarChart,

  PolarGrid,

  PolarAngleAxis,

  PolarRadiusAxis,

  Radar,

  ResponsiveContainer,

} from "recharts";



import "./App.css";





function App() {

  // =========================================================

  // 공통 상태

  // =========================================================



  const [currentPage, setCurrentPage] = useState("chat");

  const [darkMode, setDarkMode] = useState(true);

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);



  const [selectedCert, setSelectedCert] =

    useState("정보처리기사");





  // =========================================================

  // 사용자 페이지 샘플 학습 데이터

  // 나중에 API와 연결 예정

  // =========================================================



  const dashboardData = {

    정보처리기사: {

      radar: [

        {

          subject: "데이터베이스 구축",

          score: 35,

        },

        {

          subject: "정보시스템 구축 관리",

          score: 45,

        },

        {

          subject: "프로그래밍 언어 활용",

          score: 70,

        },

        {

          subject: "소프트웨어 설계",

          score: 55,

        },

        {

          subject: "소프트웨어 개발",

          score: 50,

        },

      ],



      weakest: "데이터베이스 구축",



      todaySolved: 12,

      dailyGoal: 50,



      totalSolved: 38,

      correctSolved: 20,

      accuracy: 52.6,

    },



    "공인중개사 1차": {

      radar: [

        {

          subject: "부동산학개론",

          score: 58,

        },

        {

          subject: "민법총칙",

          score: 42,

        },

        {

          subject: "물권법",

          score: 65,

        },

        {

          subject: "계약법",

          score: 52,

        },

        {

          subject: "민사특별법",

          score: 47,

        },

      ],



      weakest: "민법총칙",



      todaySolved: 8,

      dailyGoal: 50,



      totalSolved: 24,

      correctSolved: 14,

      accuracy: 58.3,

    },

  };





  const currentDashboard =

    dashboardData[selectedCert];





  const goalRate = Math.min(

    Math.round(

      (

        currentDashboard.todaySolved /

        currentDashboard.dailyGoal

      ) * 100

    ),

    100

  );





  // =========================================================

  // AI 채팅 상태

  // =========================================================



  const [messages, setMessages] = useState([]);

  const [input, setInput] = useState("");



  const [answerLength, setAnswerLength] =

    useState("medium");





  // =========================================================

  // 문제풀이 상태

  // =========================================================



  const [quiz, setQuiz] = useState(null);



  const [quizLoading, setQuizLoading] =

    useState(false);



  const [selectedAnswer, setSelectedAnswer] =

    useState(null);



  const [quizResult, setQuizResult] =

    useState(null);



  const [quizError, setQuizError] =

    useState("");





  // =========================================================

  // AI 채팅

  // =========================================================



  const sendMessage = async () => {

    const text = input.trim();



    if (!text) return;



    const userMessage = {

      id: `${Date.now()}-user`,

      role: "user",

      content: text,

    };



    setMessages((prev) => [

      ...prev,

      userMessage,

    ]);



    setInput("");



    try {

      const response = await fetch(

        "http://localhost:8000/api/chat",

        {

          method: "POST",



          headers: {

            "Content-Type":

              "application/json",

          },



          body: JSON.stringify({

            message: text,



            // 현재 백엔드는 정보처리기사(EIP)에 연결

            cert: "EIP",



            answer_length:

              answerLength,

          }),

        }

      );



      if (!response.ok) {

        throw new Error(

          `API 요청 실패: ${response.status}`

        );

      }



      const data =

        await response.json();



      const assistantMessage = {

        id: `${Date.now()}-assistant`,

        role: "assistant",

        content: data.answer,

      };



      setMessages((prev) => [

        ...prev,

        assistantMessage,

      ]);

    } catch (error) {

      console.error(

        "AI Tutor API 오류:",

        error

      );



      const errorMessage = {

        id: `${Date.now()}-error`,

        role: "assistant",



        content:

          "AI 튜터 서버와 연결하지 못했습니다. FastAPI 서버가 실행 중인지 확인해주세요.",

      };



      setMessages((prev) => [

        ...prev,

        errorMessage,

      ]);

    }

  };





  const handleKeyDown = (e) => {

    if (

      e.key === "Enter" &&

      !e.shiftKey

    ) {

      e.preventDefault();

      sendMessage();

    }

  };





  const newChat = () => {

    setCurrentPage("chat");



    setMessages([]);

    setInput("");

  };





  // =========================================================

  // 문제 생성

  // =========================================================



  const loadQuiz = async () => {

    setCurrentPage("quiz");



    setQuizLoading(true);



    setQuiz(null);

    setQuizResult(null);

    setSelectedAnswer(null);

    setQuizError("");



    try {

      const response = await fetch(

        "http://localhost:8000/api/quiz",

        {

          method: "POST",



          headers: {

            "Content-Type":

              "application/json",

          },



          body: JSON.stringify({

            // 현재 백엔드는 정보처리기사(EIP)에 연결

            cert: "EIP",

          }),

        }

      );



      if (!response.ok) {

        throw new Error(

          `문제 생성 실패: ${response.status}`

        );

      }



      const data =

        await response.json();



      setQuiz(data);

    } catch (error) {

      console.error(

        "문제 생성 오류:",

        error

      );



      setQuizError(

        "문제를 생성하지 못했습니다. FastAPI 서버와 AI 엔진 상태를 확인해주세요."

      );

    } finally {

      setQuizLoading(false);

    }

  };





  // =========================================================

  // 정답 제출

  // =========================================================



  const submitQuiz = async () => {

    if (

      !quiz ||

      selectedAnswer === null

    ) {

      return;

    }



    try {

      const response = await fetch(

        "http://localhost:8000/api/quiz/submit",

        {

          method: "POST",



          headers: {

            "Content-Type":

              "application/json",

          },



          body: JSON.stringify({

            quiz_id:

              quiz.quiz_id,



            selected_answer:

              selectedAnswer,

          }),

        }

      );



      if (!response.ok) {

        throw new Error(

          `채점 실패: ${response.status}`

        );

      }



      const data =

        await response.json();



      setQuizResult(data);

    } catch (error) {

      console.error(

        "채점 오류:",

        error

      );



      setQuizError(

        "정답을 채점하지 못했습니다."

      );

    }

  };





  // =========================================================

  // 준비 중 메뉴

  // =========================================================



  const showComingSoon = (

    feature

  ) => {

    alert(

      `${feature} 기능은 React로 이전 중입니다.`

    );

  };





  return (

    <div

      className={`app ${

        darkMode

          ? "dark"

          : "light"

      }`}

    >



      {/* =====================================================

          Sidebar

      ===================================================== */}



      <aside

        className={`sidebar ${

          sidebarCollapsed

            ? "collapsed"

            : ""

        }`}

      >



        <div className="sidebar-top">



          <div className="sidebar-header">



            {!sidebarCollapsed && (

              <div className="logo">

                AI Tutor

              </div>

            )}



            <button

              type="button"

              className="sidebar-toggle"



              onClick={() =>

                setSidebarCollapsed(

                  (prev) => !prev

                )

              }



              title={

                sidebarCollapsed

                  ? "사이드바 펼치기"

                  : "사이드바 접기"

              }

            >

              {sidebarCollapsed

                ? "›"

                : "‹"}

            </button>



          </div>





          {/* 새 대화 */}



          <button

            type="button"

            className="new-chat-button"

            onClick={newChat}

          >

            <span>＋</span>



            {!sidebarCollapsed && (

              <span>

                새 대화

              </span>

            )}

          </button>





          {/* 메뉴 */}



          <nav className="menu">



            <button

              type="button"



              className={`menu-item ${

                currentPage === "chat"

                  ? "active"

                  : ""

              }`}



              onClick={() =>

                setCurrentPage("chat")

              }

            >

              <span className="menu-icon">

                ◉

              </span>



              {!sidebarCollapsed && (

                <span>

                  AI 튜터

                </span>

              )}

            </button>





            <button

              type="button"



              className={`menu-item ${

                currentPage === "quiz"

                  ? "active"

                  : ""

              }`}



              onClick={loadQuiz}

            >

              <span className="menu-icon">

                □

              </span>



              {!sidebarCollapsed && (

                <span>

                  문제 풀이

                </span>

              )}

            </button>





            <button

              type="button"

              className="menu-item"



              onClick={() =>

                showComingSoon(

                  "취약점 학습"

                )

              }

            >

              <span className="menu-icon">

                ◇

              </span>



              {!sidebarCollapsed && (

                <span>

                  취약점 학습

                </span>

              )}

            </button>





            <button

              type="button"

              className="menu-item"



              onClick={() =>

                showComingSoon(

                  "모의고사"

                )

              }

            >

              <span className="menu-icon">

                △

              </span>



              {!sidebarCollapsed && (

                <span>

                  모의고사

                </span>

              )}

            </button>





            <button

              type="button"

              className="menu-item"



              onClick={() =>

                showComingSoon(

                  "오답노트"

                )

              }

            >

              <span className="menu-icon">

                ○

              </span>



              {!sidebarCollapsed && (

                <span>

                  오답노트

                </span>

              )}

            </button>



          </nav>

        </div>





        {/* 사용자 */}



        <div className="sidebar-bottom">



          <button

            type="button"



            className={`sidebar-user-button ${

              currentPage === "profile"

                ? "active"

                : ""

            }`}



            onClick={() =>

              setCurrentPage(

                "profile"

              )

            }



            title="나의 학습 현황"

          >



            <div className="profile-circle">

              U

            </div>



            {!sidebarCollapsed && (

              <div className="profile-info">



                <div className="profile-name">

                  서정우

                </div>



                <div className="profile-subtitle">

                  {selectedCert}

                </div>



              </div>

            )}



          </button>



        </div>



      </aside>





      {/* =====================================================

          Main

      ===================================================== */}



      <main className="main">



        {/* Header */}



        <header className="header">



          <div className="header-title">

            안양대학교 AI Tutor

          </div>



          <button

            type="button"

            className="theme-toggle"



            onClick={() =>

              setDarkMode(

                (prev) => !prev

              )

            }



            title={

              darkMode

                ? "라이트 모드"

                : "다크 모드"

            }

          >

            {darkMode

              ? "☀"

              : "☾"}

          </button>



        </header>





        {/* =================================================

            AI Tutor

        ================================================= */}



        {currentPage ===

          "chat" && (

          <>



            <section className="chat-area">



              {messages.length ===

              0 ? (



                <div className="welcome">



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



                    <button

                      type="button"



                      onClick={() =>

                        setInput(

                          "데이터베이스 정규화에 대해 설명해줘"

                        )

                      }

                    >

                      데이터베이스 정규화 설명

                    </button>





                    <button

                      type="button"



                      onClick={() =>

                        setInput(

                          "OSI 7계층을 쉽게 설명해줘"

                        )

                      }

                    >

                      OSI 7계층 설명

                    </button>





                    <button

                      type="button"



                      onClick={() =>

                        setInput(

                          "TCP와 UDP의 차이점을 알려줘"

                        )

                      }

                    >

                      TCP와 UDP 차이

                    </button>



                  </div>



                </div>



              ) : (



                <div className="messages">



                  {messages.map(

                    (message) => (



                      <div

                        key={message.id}



                        className={`message-row ${message.role}`}

                      >



                        {message.role ===

                          "assistant" && (



                          <div className="assistant-avatar">

                            AI

                          </div>

                        )}





                        <div className="message">



                          {message.role ===

                          "assistant" ? (



                            <ReactMarkdown

                              remarkPlugins={[

                                remarkGfm,

                              ]}

                            >

                              {message.content}

                            </ReactMarkdown>



                          ) : (

                            message.content

                          )}



                        </div>



                      </div>

                    )

                  )}



                </div>

              )}



            </section>





            {/* 채팅 입력 */}



            <div className="input-section">



              <div className="answer-length-control">



                <span className="answer-length-label">

                  답변 길이

                </span>





                <button

                  type="button"



                  className={

                    answerLength ===

                    "short"

                      ? "active"

                      : ""

                  }



                  onClick={() =>

                    setAnswerLength(

                      "short"

                    )

                  }

                >

                  간단히

                </button>





                <button

                  type="button"



                  className={

                    answerLength ===

                    "medium"

                      ? "active"

                      : ""

                  }



                  onClick={() =>

                    setAnswerLength(

                      "medium"

                    )

                  }

                >

                  보통

                </button>





                <button

                  type="button"



                  className={

                    answerLength ===

                    "long"

                      ? "active"

                      : ""

                  }



                  onClick={() =>

                    setAnswerLength(

                      "long"

                    )

                  }

                >

                  자세히

                </button>



              </div>





              <div className="input-container">



                <textarea

                  value={input}



                  onChange={(e) =>

                    setInput(

                      e.target.value

                    )

                  }



                  onKeyDown={

                    handleKeyDown

                  }



                  placeholder="AI 튜터에게 질문해보세요"



                  rows={1}

                />





                <button

                  type="button"

                  className="send-button"



                  onClick={

                    sendMessage

                  }



                  disabled={

                    !input.trim()

                  }

                >

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



          </>

        )}





        {/* =================================================
            문제 풀이
        ================================================= */}

        {currentPage === "quiz" && (
          <section className="quiz-page">
            <div className="quiz-header">
              <div>
                <span className="quiz-page-label">PRACTICE</span>
                <h1>문제 풀이</h1>
                <p>AI가 기출 데이터를 분석해 만든 변형 문제입니다.</p>
              </div>

              <button
                type="button"
                className="quiz-new-button"
                onClick={loadQuiz}
                disabled={quizLoading}
              >
                새 문제
              </button>
            </div>

            {quizLoading && (
              <div className="quiz-loading">
                <div className="quiz-loading-spinner" />
                <h3>AI가 문제를 만들고 있습니다.</h3>
                <p>출제 후 검수까지 진행하므로 잠시 시간이 걸릴 수 있습니다.</p>
              </div>
            )}

            {!quizLoading && quizError && (
              <div className="quiz-error">{quizError}</div>
            )}

            {!quizLoading && quiz && (
              <div
                className={`quiz-workspace ${
                  quizResult ? "has-result" : ""
                }`}
              >
                {/* 왼쪽: 원래 문제 */}
                <div className="quiz-card quiz-question-panel">
                  <div className="quiz-meta">
                    <span>{selectedCert}</span>
                    <span>{quiz.topic_label}</span>
                  </div>

                  <h2 className="quiz-question">
                    Q. {quiz.question}
                  </h2>

                  {quiz.code_block && (
                    <pre className="quiz-code">
                      <code>{quiz.code_block}</code>
                    </pre>
                  )}

                  {quiz.table_data && (
                    <div className="quiz-table">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {quiz.table_data}
                      </ReactMarkdown>
                    </div>
                  )}

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
                        } else if (
                          number === selectedAnswer &&
                          !quizResult.is_correct
                        ) {
                          optionClass += " incorrect";
                        }
                      }

                      return (
                        <button
                          type="button"
                          key={number}
                          className={optionClass}
                          disabled={!!quizResult}
                          onClick={() => setSelectedAnswer(number)}
                        >
                          <span className="quiz-option-number">{number}</span>
                          <span>
                            {option.replace(/^\s*\d+\s*[).:-]?\s*/, "")}
                          </span>
                        </button>
                      );
                    })}
                  </div>

                  {!quizResult && (
                    <div className="quiz-submit-area">
                      <button
                        type="button"
                        className="quiz-submit-button"
                        disabled={selectedAnswer === null}
                        onClick={submitQuiz}
                      >
                        정답 제출
                      </button>
                    </div>
                  )}

                  {quizResult && (
                    <div className="quiz-inline-result">
                      <div
                        className={`quiz-result-title ${
                          quizResult.is_correct ? "success" : "wrong"
                        }`}
                      >
                        {quizResult.is_correct ? "정답입니다!" : "오답입니다."}
                      </div>

                      {!quizResult.is_correct && (
                        <p className="quiz-correct-answer">
                          정답은 <strong>{quizResult.correct_answer}번</strong>입니다.
                        </p>
                      )}
                    </div>
                  )}
                </div>

                {/* 오른쪽: 답 제출 후 AI 해설 */}
                {quizResult && (
                  <aside className="quiz-solution-panel">
                    <div className="quiz-solution-header">
                      <span className="quiz-solution-label">AI SOLUTION</span>
                      <h2>AI 해설</h2>
                      <p>문제와 비교하면서 해설을 확인해보세요.</p>
                    </div>

                    <div className="quiz-explanation quiz-explanation-side">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {quizResult.explanation}
                      </ReactMarkdown>
                    </div>

                    <div className="quiz-solution-actions">
                      <button
                        type="button"
                        className="quiz-next-button"
                        onClick={loadQuiz}
                      >
                        다음 문제
                      </button>
                    </div>
                  </aside>
                )}
              </div>
            )}
          </section>
        )}


        {/* =================================================

            사용자 / 학습 현황

        ================================================= */}



        {currentPage ===

          "profile" && (



          <section className="profile-page">



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

                <button
                  type="button"
                  className="learning-reset-button"
                  onClick={() =>
                    alert("학습 데이터 초기화 기능은 추후 연결 예정입니다.")
                  }
                >
                  학습 데이터 초기화
                </button>
              </div>


              {/* =============================================

                  취약점 분석

              ============================================= */}



              <section className="dashboard-card weakness-card">



                <div className="dashboard-card-header">



                  <div>



                    <span className="dashboard-small-label">

                      LEARNING ANALYSIS

                    </span>



                    <h2>

                      나의 학습 취약점 분석

                    </h2>



                  </div>



                </div>





                <div className="radar-chart-wrapper">



                  <ResponsiveContainer

                    width="100%"

                    height="100%"

                  >



                    <RadarChart

                      data={

                        currentDashboard.radar

                      }



                      outerRadius="68%"

                    >



                      <PolarGrid

                        stroke={

                          darkMode

                            ? "#464646"

                            : "#d4d4d4"

                        }

                      />





                      <PolarAngleAxis

                        dataKey="subject"



                        tick={{

                          fill:

                            darkMode

                              ? "#c6c6c6"

                              : "#444444",



                          fontSize: 12,

                        }}

                      />





                      <PolarRadiusAxis

                        angle={90}



                        domain={[

                          0,

                          100,

                        ]}



                        tick={{

                          fill:

                            darkMode

                              ? "#777777"

                              : "#888888",



                          fontSize: 10,

                        }}

                      />





                      <Radar

                        name="정답률"

                        dataKey="score"



                        stroke="#4da3ff"

                        fill="#4da3ff"



                        fillOpacity={0.32}



                        strokeWidth={2}

                      />



                    </RadarChart>



                  </ResponsiveContainer>



                </div>





                <div className="weakness-analysis">



                  <span className="analysis-icon">

                    💡

                  </span>



                  <span>

                    분석 결과: 현재{" "}



                    <strong>

                      {

                        currentDashboard.weakest

                      }

                    </strong>



                    {" "}과목이 가장 취약합니다.

                    집중 공부가 필요합니다.

                  </span>



                </div>



              </section>





              {/* =============================================

                  목표 / 전체 현황

              ============================================= */}



              <div className="learning-summary-grid">



                {/* 오늘의 목표 */}



                <section className="dashboard-card summary-card">



                  <div className="summary-title">



                    <span className="summary-icon">

                      🎯

                    </span>



                    <h2>

                      오늘의 목표 달성률

                    </h2>



                  </div>





                  <div className="summary-content">



                    <span className="summary-label">

                      오늘 풀이 수

                    </span>





                    <div className="summary-big-number">



                      {

                        currentDashboard.todaySolved

                      }



                      <span>

                        {" "}/{" "}



                        {

                          currentDashboard.dailyGoal

                        }

                      </span>



                    </div>





                    <div className="goal-progress">



                      <div

                        className="goal-progress-bar"



                        style={{

                          width:

                            `${goalRate}%`,

                        }}

                      />



                    </div>





                    <div className="summary-caption">



                      현재 목표 달성률:{" "}



                      {goalRate}%



                    </div>



                  </div>



                </section>





                {/* 전체 학습 현황 */}



                <section className="dashboard-card summary-card">



                  <div className="summary-title">



                    <span className="summary-icon">

                      📊

                    </span>



                    <h2>

                      전체 학습 현황

                    </h2>



                  </div>





                  <div className="total-stats">



                    <div className="total-stat-item">



                      <span>

                        📝 푼 문제

                      </span>



                      <strong>

                        {

                          currentDashboard.totalSolved

                        }



                        <small>

                          개

                        </small>

                      </strong>



                    </div>





                    <div className="total-stat-item">



                      <span>

                        🎯 맞춘 문제

                      </span>



                      <strong>

                        {

                          currentDashboard.correctSolved

                        }



                        <small>

                          개

                        </small>

                      </strong>



                    </div>





                    <div className="total-stat-item">



                      <span>

                        🔥 정답률

                      </span>



                      <strong>

                        {

                          currentDashboard.accuracy

                        }



                        <small>

                          %

                        </small>

                      </strong>



                    </div>



                  </div>



                </section>



              </div>





              {/* =============================================

                  자격증 설정

              ============================================= */}



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



                  <button

                    type="button"



                    className={`cert-card ${

                      selectedCert ===

                      "정보처리기사"

                        ? "selected"

                        : ""

                    }`}



                    onClick={() =>

                      setSelectedCert(

                        "정보처리기사"

                      )

                    }

                  >



                    <div className="cert-card-top">



                      <div className="cert-icon">

                        IT

                      </div>



                      {selectedCert ===

                        "정보처리기사" && (



                        <div className="cert-check">

                          ✓

                        </div>

                      )}



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



                  <button

                    type="button"



                    className={`cert-card ${

                      selectedCert ===

                      "공인중개사 1차"

                        ? "selected"

                        : ""

                    }`}



                    onClick={() =>

                      setSelectedCert(

                        "공인중개사 1차"

                      )

                    }

                  >



                    <div className="cert-card-top">



                      <div className="cert-icon">

                        RE

                      </div>



                      {selectedCert ===

                        "공인중개사 1차" && (



                        <div className="cert-check">

                          ✓

                        </div>

                      )}



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



          </section>

        )}



      </main>



    </div>

  );

}





export default App;