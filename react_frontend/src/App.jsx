import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from "recharts";
import "./App.css";
function App() {
  // =========================================================
  // 공통 상태
  // =========================================================
  const [currentPage, setCurrentPage] = useState("chat");
  const [darkMode, setDarkMode] = useState(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [selectedCert, setSelectedCert] = useState("정보처리기사");
  // =========================================================
  // 사용자 페이지 샘플 학습 데이터
  // 나중에 API와 연결 예정
  // =========================================================
  const dashboardData = {
    정보처리기사: {
      radar: [{
        subject: "데이터베이스 구축",
        score: 35
      }, {
        subject: "정보시스템 구축 관리",
        score: 45
      }, {
        subject: "프로그래밍 언어 활용",
        score: 70
      }, {
        subject: "소프트웨어 설계",
        score: 55
      }, {
        subject: "소프트웨어 개발",
        score: 50
      }],
      weakest: "데이터베이스 구축",
      todaySolved: 12,
      dailyGoal: 50,
      totalSolved: 38,
      correctSolved: 20,
      accuracy: 52.6
    },
    "공인중개사 1차": {
      radar: [{
        subject: "부동산학개론",
        score: 58
      }, {
        subject: "민법총칙",
        score: 42
      }, {
        subject: "물권법",
        score: 65
      }, {
        subject: "계약법",
        score: 52
      }, {
        subject: "민사특별법",
        score: 47
      }],
      weakest: "민법총칙",
      todaySolved: 8,
      dailyGoal: 50,
      totalSolved: 24,
      correctSolved: 14,
      accuracy: 58.3
    }
  };
  const currentDashboard = dashboardData[selectedCert];
  const goalRate = Math.min(Math.round(currentDashboard.todaySolved / currentDashboard.dailyGoal * 100), 100);
  // =========================================================
  // =========================================================
  // 취약점 학습 샘플 데이터
  // 현재는 프론트 화면 시연용이며, 추후 API/DB 데이터로 교체 예정
  // =========================================================
  const weaknessData = {
    "정보처리기사": {
      focus: "데이터베이스 구축",
      accuracy: 35,
      description: "정규화, 트랜잭션, 병행 제어 영역의 오답률이 높습니다. 핵심 개념을 다시 확인한 뒤 집중 문제를 풀어보세요.",
      ranking: [{
        topic: "데이터베이스 구축",
        accuracy: 35
      }, {
        topic: "정보시스템 구축 관리",
        accuracy: 45
      }, {
        topic: "소프트웨어 개발",
        accuracy: 50
      }],
      question: "다음 중 데이터베이스 트랜잭션의 ACID 특성에 대한 설명으로 옳은 것은?",
      options: ["Atomicity는 트랜잭션의 일부 연산만 성공해도 결과를 반영하는 성질이다.", "Consistency는 트랜잭션 수행 전후에 데이터베이스가 일관된 상태를 유지하는 성질이다.", "Isolation은 완료된 트랜잭션의 결과가 영구적으로 저장되는 성질이다.", "Durability는 동시에 실행되는 트랜잭션이 서로 간섭하지 않는 성질이다."],
      correctAnswer: 2,
      explanation: "Consistency(일관성)는 트랜잭션 실행 전후에도 데이터베이스가 정의된 규칙과 제약조건을 만족하는 일관된 상태를 유지해야 한다는 특성입니다."
    },
    "공인중개사 1차": {
      focus: "민법총칙",
      accuracy: 42,
      description: "의사표시와 법률행위 영역의 정답률이 낮습니다. 자주 혼동되는 요건을 중심으로 다시 학습해보세요.",
      ranking: [{
        topic: "민법총칙",
        accuracy: 42
      }, {
        topic: "민사특별법",
        accuracy: 47
      }, {
        topic: "계약법",
        accuracy: 52
      }],
      question: "다음 중 민법상 의사표시에 관한 설명으로 가장 적절한 것은?",
      options: ["모든 착오에 의한 의사표시는 언제나 무효이다.", "진의 아닌 의사표시는 상대방이 그 사실을 알았거나 알 수 있었던 경우 무효가 될 수 있다.", "사기에 의한 의사표시는 어떠한 경우에도 취소할 수 없다.", "강박에 의한 의사표시는 처음부터 당연히 무효이다."],
      correctAnswer: 2,
      explanation: "진의 아닌 의사표시는 원칙적으로 유효하지만, 상대방이 표의자의 진의 아님을 알았거나 알 수 있었던 경우에는 무효가 될 수 있습니다."
    }
  };
  const currentWeakness = weaknessData[selectedCert];
  // =========================================================
  // 오답노트 / 요약 노트 샘플 데이터
  // 현재는 프론트 화면 시연용이며, 추후 취약점 분석 API와 연결 예정
  // =========================================================
  const summaryNoteData = {
    "정보처리기사": [{
      rank: 1,
      topic: "데이터베이스 구축",
      accuracy: 35,
      concepts: ["정규화", "트랜잭션", "병행 제어"],
      mistakes: ["정규화 단계별 목적과 이상 현상을 함께 구분하기", "ACID의 Atomicity / Consistency / Isolation / Durability 의미 구분", "락(Lock)과 병행 제어의 역할을 혼동하지 않기"],
      memory: "ACID = 원자성 · 일관성 · 고립성 · 지속성. 정규화는 데이터 중복과 이상 현상을 줄이기 위한 과정입니다."
    }, {
      rank: 2,
      topic: "정보시스템 구축 관리",
      accuracy: 45,
      concepts: ["보안 공격", "네트워크", "소프트웨어 보안"],
      mistakes: ["공격 기법의 특징과 대응 방법 연결하기", "네트워크 장비와 프로토콜 역할 구분하기", "접근 통제와 인증 개념을 섞지 않기"],
      memory: "보안 문제는 공격 기법의 정의만 외우기보다 '공격 방식 → 영향 → 대응 방법' 순서로 묶어서 기억합니다."
    }, {
      rank: 3,
      topic: "소프트웨어 개발",
      accuracy: 50,
      concepts: ["테스트", "자료구조", "인터페이스 구현"],
      mistakes: ["테스트 단계별 목적과 수행 주체 구분하기", "자료구조별 탐색·삽입·삭제 특성 비교하기", "인터페이스 구현 절차와 검증 항목 확인하기"],
      memory: "단위 → 통합 → 시스템 → 인수 테스트 순서를 기억하고, 각 단계의 목적을 문제 문장과 연결해서 판단합니다."
    }],
    "공인중개사 1차": [{
      rank: 1,
      topic: "민법총칙",
      accuracy: 42,
      concepts: ["의사표시", "법률행위", "무효와 취소"],
      mistakes: ["진의 아닌 의사표시와 통정허위표시 구분하기", "착오·사기·강박의 취소 요건 비교하기", "무효와 취소의 법적 효과를 혼동하지 않기"],
      memory: "의사표시 문제는 '당사자의 진의 → 상대방의 인식 → 법률효과' 순서로 조건을 확인합니다."
    }, {
      rank: 2,
      topic: "민사특별법",
      accuracy: 47,
      concepts: ["주택임대차", "상가임대차", "집합건물"],
      mistakes: ["대항력과 우선변제권의 요건 구분하기", "주택·상가 임대차의 적용 범위를 비교하기", "보호 대상과 효력 발생 시점을 확인하기"],
      memory: "임대차 문제는 '적용 대상 → 요건 → 효력 발생 시점' 세 단계로 정리하면 실수를 줄일 수 있습니다."
    }, {
      rank: 3,
      topic: "계약법",
      accuracy: 52,
      concepts: ["계약 성립", "동시이행", "해제와 해지"],
      mistakes: ["청약과 승낙의 효력 발생 시점 구분하기", "동시이행항변권의 성립 요건 확인하기", "해제와 해지의 효과를 구분하기"],
      memory: "계약 문제는 성립 → 이행 → 불이행 → 종료의 흐름으로 정리하고 각 단계의 요건을 확인합니다."
    }]
  };
  const currentSummaryNotes = summaryNoteData[selectedCert];
  // =========================================================
  // 모의고사 샘플 데이터
  // 현재는 프론트 화면 시연용이며, 추후 실제 문제 API로 교체 예정
  // =========================================================
  const mockExamData = {
    정보처리기사: {
      subjects: ["소프트웨어 설계", "소프트웨어 개발", "데이터베이스 구축", "프로그래밍 언어 활용", "정보시스템 구축 관리"],
      questions: [{
        id: "eip-1",
        subject: "소프트웨어 설계",
        question: "UML 시퀀스 다이어그램의 주된 표현 대상으로 가장 적절한 것은?",
        options: ["시스템의 물리적 배치 구조", "객체 간 메시지 전달을 시간 순서에 따라 표현", "데이터베이스 테이블 간 관계", "프로그램의 소스 코드 복잡도"],
        answer: 2,
        explanation: "시퀀스 다이어그램은 객체 사이에서 주고받는 메시지와 상호작용을 시간 흐름에 따라 표현하는 UML 다이어그램입니다."
      }, {
        id: "eip-2",
        subject: "소프트웨어 설계",
        question: "좋은 소프트웨어 모듈 설계에 대한 설명으로 가장 적절한 것은?",
        options: ["응집도는 낮고 결합도는 높게 설계한다.", "응집도와 결합도를 모두 높게 설계한다.", "응집도는 높고 결합도는 낮게 설계한다.", "응집도와 결합도를 모두 낮게 설계한다."],
        answer: 3,
        explanation: "모듈 내부의 관련성인 응집도는 높이고, 모듈 사이의 의존성인 결합도는 낮추는 것이 일반적으로 바람직합니다."
      }, {
        id: "eip-3",
        subject: "소프트웨어 개발",
        question: "화이트박스 테스트의 특징으로 옳은 것은?",
        options: ["프로그램 내부 구조와 논리를 고려하여 테스트한다.", "사용자 요구사항만으로 테스트 케이스를 만든다.", "소스 코드를 전혀 확인하지 않는다.", "시스템 설치 환경만 검증한다."],
        answer: 1,
        explanation: "화이트박스 테스트는 프로그램 내부의 제어 구조, 경로, 조건 등 구현 논리를 기반으로 테스트합니다."
      }, {
        id: "eip-4",
        subject: "소프트웨어 개발",
        question: "스택(Stack) 자료구조의 기본 동작 방식은?",
        options: ["FIFO", "LIFO", "무작위 접근만 가능", "우선순위가 낮은 데이터부터 삭제"],
        answer: 2,
        explanation: "스택은 가장 나중에 들어온 데이터가 가장 먼저 나가는 LIFO(Last In First Out) 방식의 자료구조입니다."
      }, {
        id: "eip-5",
        subject: "데이터베이스 구축",
        question: "데이터베이스 정규화의 주요 목적으로 가장 적절한 것은?",
        options: ["데이터 중복과 이상 현상을 줄인다.", "모든 테이블을 하나로 합친다.", "검색 속도를 위해 무조건 중복을 증가시킨다.", "기본키를 제거한다."],
        answer: 1,
        explanation: "정규화는 데이터 중복을 줄이고 삽입·삭제·갱신 이상을 방지하도록 릴레이션을 구조화하는 과정입니다."
      }, {
        id: "eip-6",
        subject: "데이터베이스 구축",
        question: "트랜잭션의 ACID 특성 중 일관성(Consistency)에 대한 설명은?",
        options: ["트랜잭션의 연산은 전부 수행되거나 전혀 수행되지 않아야 한다.", "동시에 수행되는 트랜잭션은 서로의 중간 결과에 영향을 주지 않아야 한다.", "트랜잭션 수행 전후에도 데이터베이스가 일관된 상태를 유지해야 한다.", "완료된 트랜잭션 결과는 영구적으로 보존되어야 한다."],
        answer: 3,
        explanation: "Consistency는 트랜잭션 수행 전과 후에 데이터베이스가 정의된 제약조건을 만족하는 일관된 상태를 유지하는 특성입니다."
      }, {
        id: "eip-7",
        subject: "프로그래밍 언어 활용",
        question: "다음 중 TCP의 특징으로 옳은 것은?",
        options: ["비연결형 전송만 지원한다.", "전송 순서와 신뢰성을 보장하지 않는다.", "연결 지향적이며 신뢰성 있는 전송을 제공한다.", "항상 UDP보다 헤더가 작다."],
        answer: 3,
        explanation: "TCP는 연결 지향형 프로토콜로 순서 제어, 오류 제어 등을 통해 신뢰성 있는 데이터 전송을 제공합니다."
      }, {
        id: "eip-8",
        subject: "프로그래밍 언어 활용",
        question: "후입선출(LIFO) 구조를 직접적으로 활용하는 알고리즘에 가장 가까운 것은?",
        options: ["함수 호출의 실행 관리", "프린터 대기열", "라운드 로빈 스케줄링", "선착순 민원 처리"],
        answer: 1,
        explanation: "함수 호출은 호출 정보를 스택에 저장하고 가장 최근에 호출된 함수부터 복귀하므로 LIFO 구조를 사용합니다."
      }, {
        id: "eip-9",
        subject: "정보시스템 구축 관리",
        question: "정보보안에서 인증(Authentication)의 의미로 가장 적절한 것은?",
        options: ["사용자가 누구인지 확인하는 과정", "사용자에게 모든 권한을 부여하는 과정", "데이터를 무조건 공개하는 과정", "네트워크 장비를 제거하는 과정"],
        answer: 1,
        explanation: "인증은 접근을 요청한 사용자가 주장하는 신원을 실제로 확인하는 과정입니다."
      }, {
        id: "eip-10",
        subject: "정보시스템 구축 관리",
        question: "다음 중 네트워크에서 서로 다른 네트워크 사이의 패킷 전달 경로를 결정하는 장비는?",
        options: ["허브", "리피터", "라우터", "NIC"],
        answer: 3,
        explanation: "라우터는 네트워크 계층에서 서로 다른 네트워크 사이의 패킷 전달 경로를 선택하고 전달합니다."
      }]
    },
    "공인중개사 1차": {
      subjects: ["부동산학개론", "민법총칙", "물권법", "계약법", "민사특별법"],
      questions: [{
        id: "re-1",
        subject: "부동산학개론",
        question: "다른 조건이 일정할 때 일반적인 수요 법칙에 대한 설명으로 가장 적절한 것은?",
        options: ["가격이 상승하면 수요량도 항상 증가한다.", "가격이 상승하면 수요량은 감소하는 경향이 있다.", "가격과 수요량은 아무 관계가 없다.", "가격이 하락하면 수요량도 반드시 감소한다."],
        answer: 2,
        explanation: "일반적인 수요 법칙에서는 다른 조건이 일정할 때 가격과 수요량이 반대 방향으로 움직이는 경향이 있습니다."
      }, {
        id: "re-2",
        subject: "부동산학개론",
        question: "수익환원법에서 다른 조건이 동일할 때 환원이율이 상승하면 부동산 가치에는 일반적으로 어떤 영향이 있는가?",
        options: ["가치가 상승한다.", "가치가 하락한다.", "가치가 반드시 동일하다.", "환원이율과 가치는 무관하다."],
        answer: 2,
        explanation: "다른 조건이 일정하면 수익을 더 높은 환원이율로 자본환원할수록 산정되는 가치는 낮아지는 관계가 나타납니다."
      }, {
        id: "re-3",
        subject: "민법총칙",
        question: "진의 아닌 의사표시에 대한 설명으로 가장 적절한 것은?",
        options: ["언제나 당연히 무효이다.", "원칙적으로 효력이 있으나 상대방이 진의 아님을 알았거나 알 수 있었던 경우에는 무효가 될 수 있다.", "언제나 취소할 수 없다.", "상대방의 인식 여부는 전혀 고려하지 않는다."],
        answer: 2,
        explanation: "진의 아닌 의사표시는 원칙적으로 유효하지만, 상대방이 표의자의 진의 아님을 알았거나 알 수 있었던 경우에는 무효가 될 수 있습니다."
      }, {
        id: "re-4",
        subject: "민법총칙",
        question: "사기 또는 강박에 의한 의사표시의 법적 효과로 가장 적절한 것은?",
        options: ["일반적으로 취소할 수 있다.", "언제나 처음부터 무효이다.", "어떠한 경우에도 유효를 주장할 수 없다.", "법률행위와 관계없이 형사처벌만 문제된다."],
        answer: 1,
        explanation: "민법상 사기나 강박에 의한 의사표시는 일정한 요건 아래 취소할 수 있는 의사표시로 다뤄집니다."
      }, {
        id: "re-5",
        subject: "물권법",
        question: "부동산에 관한 법률행위로 인한 물권변동의 원칙적인 효력요건은?",
        options: ["점유만 있으면 된다.", "등기가 필요하다.", "구두 합의만 있으면 된다.", "공증만 있으면 된다."],
        answer: 2,
        explanation: "부동산에 관한 법률행위로 인한 물권의 득실변경은 원칙적으로 등기해야 효력이 발생합니다."
      }, {
        id: "re-6",
        subject: "물권법",
        question: "소유권에 대한 설명으로 가장 적절한 것은?",
        options: ["법률의 범위 내에서 목적물을 사용·수익·처분할 수 있는 권리이다.", "점유와 항상 동일한 개념이다.", "채권자만 가질 수 있는 권리이다.", "부동산에는 인정되지 않는다."],
        answer: 1,
        explanation: "소유자는 법률의 범위 내에서 소유물을 사용하고 수익하며 처분할 수 있습니다."
      }, {
        id: "re-7",
        subject: "계약법",
        question: "쌍무계약에서 상대방이 채무를 이행할 때까지 자신의 채무 이행을 거절할 수 있는 권리는?",
        options: ["취소권", "동시이행의 항변권", "대리권", "상계권"],
        answer: 2,
        explanation: "쌍무계약의 서로 대가적인 채무가 이행기에 있는 경우 일정한 요건 아래 동시이행의 항변권이 문제됩니다."
      }, {
        id: "re-8",
        subject: "계약법",
        question: "계약금이 해약금으로 기능하는 경우, 당사자 일방이 이행에 착수하기 전 해제에 대한 설명으로 가장 적절한 것은?",
        options: ["계약금을 준 사람은 계약금을 포기하여 해제할 수 있다.", "계약금을 받은 사람은 같은 금액만 반환하면 된다.", "어떠한 경우에도 해제할 수 없다.", "반드시 법원의 허가가 있어야 한다."],
        answer: 1,
        explanation: "해약금 약정으로 보는 경우 이행 착수 전까지 교부자는 계약금을 포기하는 방식으로 해제할 수 있고, 수령자는 그 배액을 상환하는 방식이 문제됩니다."
      }, {
        id: "re-9",
        subject: "민사특별법",
        question: "주택임대차보호제도의 취지로 가장 적절한 것은?",
        options: ["주거용 건물 임차인의 주거생활 안정을 보호한다.", "모든 상가 계약을 무효로 한다.", "임대인의 소유권을 소멸시킨다.", "토지거래를 전면 금지한다."],
        answer: 1,
        explanation: "주택임대차 관련 제도는 주거용 건물의 임차인을 보호하고 주거생활의 안정을 도모하는 취지를 가집니다."
      }, {
        id: "re-10",
        subject: "민사특별법",
        question: "집합건물에서 구분소유의 대상이 되는 전유부분에 요구되는 성격으로 가장 적절한 것은?",
        options: ["구조상·이용상 독립성이 인정되는 부분", "반드시 건물 전체", "공용부분만 가능", "토지만 가능"],
        answer: 1,
        explanation: "집합건물의 전유부분은 구조상 및 이용상 독립성이 인정되는 건물 부분을 전제로 합니다."
      }]
    }
  };
  const currentMockExam = mockExamData[selectedCert];
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
  // =========================================================
  // =========================================================
  // 취약점 학습 상태
  // =========================================================
  const [weaknessAnswer, setWeaknessAnswer] = useState(null);
  const [weaknessSubmitted, setWeaknessSubmitted] = useState(false);
  // =========================================================
  // 모의고사 상태
  // =========================================================
  const [mockStage, setMockStage] = useState("setup");
  const [mockQuestionCount, setMockQuestionCount] = useState(10);
  const [mockSelectedSubjects, setMockSelectedSubjects] = useState(mockExamData["정보처리기사"].subjects);
  const [mockQuestions, setMockQuestions] = useState([]);
  const [mockIndex, setMockIndex] = useState(0);
  const [mockAnswers, setMockAnswers] = useState({});
  const [mockShowSubmitModal, setMockShowSubmitModal] = useState(false);
  const [mockReviewIndex, setMockReviewIndex] = useState(0);
  const mockAnsweredCount = Object.keys(mockAnswers).length;
  const mockCurrentQuestion = mockQuestions[mockIndex] || null;
  const mockCorrectCount = mockQuestions.reduce((count, question, index) => {
    return count + (mockAnswers[index] === question.answer ? 1 : 0);
  }, 0);
  const mockScore = mockQuestions.length ? Math.round(mockCorrectCount / mockQuestions.length * 100) : 0;
  const mockSubjectStats = Array.from(new Set(mockQuestions.map(question => question.subject))).map(subject => {
    const subjectQuestions = mockQuestions.map((question, index) => ({
      question,
      index
    })).filter(({
      question
    }) => question.subject === subject);
    const correct = subjectQuestions.filter(({
      question,
      index
    }) => mockAnswers[index] === question.answer).length;
    const total = subjectQuestions.length;
    return {
      subject,
      correct,
      total,
      accuracy: total ? Math.round(correct / total * 100) : 0
    };
  });
  const mockReviewQuestion = mockQuestions[mockReviewIndex] || null;
  // AI 채팅
  // =========================================================
  const sendMessage = async () => {
    const text = input.trim();
    if (!text) return;
    const userMessage = {
      id: `${Date.now()}-user`,
      role: "user",
      content: text
    };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          message: text,
          // 현재 백엔드는 정보처리기사(EIP)에 연결
          cert: "EIP",
          answer_length: answerLength
        })
      });
      if (!response.ok) {
        throw new Error(`API 요청 실패: ${response.status}`);
      }
      const data = await response.json();
      const assistantMessage = {
        id: `${Date.now()}-assistant`,
        role: "assistant",
        content: data.answer
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("AI Tutor API 오류:", error);
      const errorMessage = {
        id: `${Date.now()}-error`,
        role: "assistant",
        content: "AI 튜터 서버와 연결하지 못했습니다. FastAPI 서버가 실행 중인지 확인해주세요."
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };
  const handleKeyDown = e => {
    if (e.key === "Enter" && !e.shiftKey) {
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
      const response = await fetch("http://localhost:8000/api/quiz", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          // 현재 백엔드는 정보처리기사(EIP)에 연결
          cert: "EIP"
        })
      });
      if (!response.ok) {
        throw new Error(`문제 생성 실패: ${response.status}`);
      }
      const data = await response.json();
      setQuiz(data);
    } catch (error) {
      console.error("문제 생성 오류:", error);
      setQuizError("문제를 생성하지 못했습니다. FastAPI 서버와 AI 엔진 상태를 확인해주세요.");
    } finally {
      setQuizLoading(false);
    }
  };
  // =========================================================
  // 정답 제출
  // =========================================================
  const submitQuiz = async () => {
    if (!quiz || selectedAnswer === null) {
      return;
    }
    try {
      const response = await fetch("http://localhost:8000/api/quiz/submit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          quiz_id: quiz.quiz_id,
          selected_answer: selectedAnswer
        })
      });
      if (!response.ok) {
        throw new Error(`채점 실패: ${response.status}`);
      }
      const data = await response.json();
      setQuizResult(data);
    } catch (error) {
      console.error("채점 오류:", error);
      setQuizError("정답을 채점하지 못했습니다.");
    }
  };
  // =========================================================
  // =========================================================
  // 취약점 학습 - 프론트 샘플 동작
  // =========================================================
  const openWeaknessPage = () => {
    setCurrentPage("weakness");
    setWeaknessAnswer(null);
    setWeaknessSubmitted(false);
  };
  const submitWeaknessAnswer = () => {
    if (weaknessAnswer === null) return;
    setWeaknessSubmitted(true);
  };
  const resetWeaknessQuestion = () => {
    setWeaknessAnswer(null);
    setWeaknessSubmitted(false);
  };
  // =========================================================
  // 오답노트 / 요약 노트 - 프론트 샘플 동작
  // =========================================================
  const openWrongNotePage = () => {
    setCurrentPage("wrongnote");
  };
  const downloadSummaryNote = () => {
    const noteText = currentSummaryNotes.map(note => `# ${note.rank}. ${note.topic}

- 현재 정답률: ${note.accuracy}%

## 핵심 개념
${note.concepts.map(concept => `- ${concept}`).join("\n")}

## 자주 틀리는 포인트
${note.mistakes.map(mistake => `- ${mistake}`).join("\n")}

## 시험 직전 암기
${note.memory}
`).join("\n---\n\n");
    const markdown = `# ${selectedCert} 취약점 기반 요약 노트

현재는 React 프론트엔드 시연용 샘플 데이터입니다.

${noteText}`;
    const blob = new Blob([markdown], {
      type: "text/markdown;charset=utf-8"
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${selectedCert}_취약점_요약노트.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };
  // =========================================================
  // 모의고사 - 프론트 샘플 동작
  // =========================================================
  const openMockExamPage = () => {
    setCurrentPage("mockexam");
    setMockStage("setup");
    setMockQuestionCount(10);
    setMockSelectedSubjects([...mockExamData[selectedCert].subjects]);
    setMockQuestions([]);
    setMockIndex(0);
    setMockAnswers({});
    setMockShowSubmitModal(false);
    setMockReviewIndex(0);
  };
  const toggleMockSubject = subject => {
    setMockSelectedSubjects(prev => {
      if (prev.includes(subject)) {
        return prev.filter(item => item !== subject);
      }
      return [...prev, subject];
    });
  };
  const startMockExam = () => {
    if (mockSelectedSubjects.length === 0) return;
    const pool = currentMockExam.questions.filter(question => mockSelectedSubjects.includes(question.subject));
    if (pool.length === 0) return;
    const generatedQuestions = Array.from({
      length: mockQuestionCount
    }, (_, index) => ({
      ...pool[index % pool.length],
      instanceId: `${pool[index % pool.length].id}-${index}`
    }));
    setMockQuestions(generatedQuestions);
    setMockAnswers({});
    setMockIndex(0);
    setMockReviewIndex(0);
    setMockShowSubmitModal(false);
    setMockStage("exam");
  };
  const selectMockAnswer = answerNumber => {
    setMockAnswers(prev => ({
      ...prev,
      [mockIndex]: answerNumber
    }));
  };
  const moveMockQuestion = nextIndex => {
    if (nextIndex < 0 || nextIndex >= mockQuestions.length) return;
    setMockIndex(nextIndex);
  };
  const openMockSubmit = () => {
    setMockShowSubmitModal(true);
  };
  const submitMockExam = () => {
    const firstWrongIndex = mockQuestions.findIndex((question, index) => mockAnswers[index] !== question.answer);
    setMockReviewIndex(firstWrongIndex >= 0 ? firstWrongIndex : 0);
    setMockShowSubmitModal(false);
    setMockStage("result");
  };
  const restartMockExam = () => {
    setMockStage("setup");
    setMockQuestionCount(10);
    setMockSelectedSubjects([...currentMockExam.subjects]);
    setMockQuestions([]);
    setMockIndex(0);
    setMockAnswers({});
    setMockShowSubmitModal(false);
    setMockReviewIndex(0);
  };
  const goToWeaknessFromMock = () => {
    setCurrentPage("weakness");
    setWeaknessAnswer(null);
    setWeaknessSubmitted(false);
  };
  return <div className={`app ${darkMode ? "dark" : "light"}`}>
      {/* =====================================================
          Sidebar
       ===================================================== */}
      <aside className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
        <div className="sidebar-top">
          <div className="sidebar-header">
            {!sidebarCollapsed && <div className="logo">
                AI Tutor
              </div>}
            <button type="button" className="sidebar-toggle" onClick={() => setSidebarCollapsed(prev => !prev)} title={sidebarCollapsed ? "사이드바 펼치기" : "사이드바 접기"}>
              {sidebarCollapsed ? "›" : "‹"}
            </button>
          </div>
          {/* 새 대화 */}
          <button type="button" className="new-chat-button" onClick={newChat}>
            <span>＋</span>
            {!sidebarCollapsed && <span>
                새 대화
              </span>}
          </button>
          {/* 메뉴 */}
          <nav className="menu">
            <button type="button" className={`menu-item ${currentPage === "chat" ? "active" : ""}`} onClick={() => setCurrentPage("chat")}>
              <span className="menu-icon">
                ◉
              </span>
              {!sidebarCollapsed && <span>
                  AI 튜터
                </span>}
            </button>
            <button type="button" className={`menu-item ${currentPage === "quiz" ? "active" : ""}`} onClick={loadQuiz}>
              <span className="menu-icon">
                □
              </span>
              {!sidebarCollapsed && <span>
                  문제 풀이
                </span>}
            </button>
            <button type="button" className={`menu-item ${currentPage === "weakness" ? "active" : ""}`} onClick={openWeaknessPage}>
              <span className="menu-icon">◇</span>
              {!sidebarCollapsed && <span>취약점 학습</span>}
            </button>
            <button type="button" className={`menu-item ${currentPage === "wrongnote" ? "active" : ""}`} onClick={openWrongNotePage} title="오답노트">
              <span className="menu-icon">○</span>
              {!sidebarCollapsed && <span>오답노트</span>}
            </button>
            <button type="button" className={`menu-item ${currentPage === "mockexam" ? "active" : ""}`} onClick={openMockExamPage} title="모의고사">
              <span className="menu-icon">△</span>
              {!sidebarCollapsed && <span>모의고사</span>}
            </button>
          </nav>
        </div>
        {/* 사용자 */}
        <div className="sidebar-bottom">
          <button type="button" className={`sidebar-user-button ${currentPage === "profile" ? "active" : ""}`} onClick={() => setCurrentPage("profile")} title="나의 학습 현황">
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
          <button type="button" className="theme-toggle" onClick={() => setDarkMode(prev => !prev)} title={darkMode ? "라이트 모드" : "다크 모드"}>
            {darkMode ? "☀" : "☾"}
          </button>
        </header>
        {/* =================================================
            AI Tutor
         ================================================= */}
        {currentPage === "chat" && <>
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
                </div>}
            </section>
            {/* 채팅 입력 */}
            <div className="input-section">
              <div className="answer-length-control">
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
                <textarea value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKeyDown} placeholder="AI 튜터에게 질문해보세요" rows={1} />
                <button type="button" className="send-button" onClick={sendMessage} disabled={!input.trim()}>
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
                <p>AI가 기출 데이터를 분석해 만든 변형 문제입니다.</p>
              </div>
              <button type="button" className="quiz-new-button" onClick={loadQuiz} disabled={quizLoading}>
                새 문제
              </button>
            </div>
            {quizLoading && <div className="quiz-loading">
                <div className="quiz-loading-spinner" />
                <h3>AI가 문제를 만들고 있습니다.</h3>
                <p>출제 후 검수까지 진행하므로 잠시 시간이 걸릴 수 있습니다.</p>
              </div>}
            {!quizLoading && quizError && <div className="quiz-error">{quizError}</div>}
            {!quizLoading && quiz && <div className={`quiz-workspace ${quizResult ? "has-result" : ""}`}>
                {/* 왼쪽: 원래 문제 */}
                <div className="quiz-card quiz-question-panel">
                  <div className="quiz-meta">
                    <span>{selectedCert}</span>
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
                return <button type="button" key={number} className={optionClass} disabled={!!quizResult} onClick={() => setSelectedAnswer(number)}>
                          <span className="quiz-option-number">{number}</span>
                          <span>
                            {option.replace(/^\s*\d+\s*[).:-]?\s*/, "")}
                          </span>
                        </button>;
              })}
                  </div>
                  {!quizResult && <div className="quiz-submit-area">
                      <button type="button" className="quiz-submit-button" disabled={selectedAnswer === null} onClick={submitQuiz}>
                        정답 제출
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
                {/* 오른쪽: 답 제출 후 AI 해설 */}
                {quizResult && <aside className="quiz-solution-panel">
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
                      <button type="button" className="quiz-next-button" onClick={loadQuiz}>
                        다음 문제
                      </button>
                    </div>
                  </aside>}
              </div>}
          </section>}
        {/* =================================================
            취약점 집중 학습
         ================================================= */}
        {currentPage === "weakness" && <section className="weakness-page">
            <div className="weakness-page-content">
              <div className="weakness-page-header">
                <div>
                  <span className="weakness-page-label">WEAKNESS TRAINING</span>
                  <h1>취약점 집중 학습</h1>
                  <p>취약한 영역을 확인하고 해당 개념을 집중적으로 연습합니다.</p>
                </div>
                <span className="weakness-sample-badge">FRONTEND SAMPLE</span>
              </div>
              <div className="weakness-overview-grid">
                <section className="weakness-panel weakness-focus-panel">
                  <div className="weakness-panel-kicker">PRIORITY AREA</div>
                  <div className="weakness-focus-top">
                    <div>
                      <span className="weakness-focus-label">현재 가장 취약한 영역</span>
                      <h2>{currentWeakness.focus}</h2>
                    </div>
                    <div className="weakness-status-chip">집중 학습 필요</div>
                  </div>
                  <p className="weakness-focus-description">{currentWeakness.description}</p>
                  <div className="weakness-score-row">
                    <span>현재 정답률</span>
                    <strong>{currentWeakness.accuracy}%</strong>
                  </div>
                  <div className="weakness-progress">
                    <div className="weakness-progress-value" style={{
                  width: `${currentWeakness.accuracy}%`
                }} />
                  </div>
                </section>
                <section className="weakness-panel weakness-ranking-panel">
                  <div className="weakness-panel-kicker">WEAK SUBJECTS</div>
                  <h2>취약 영역 순위</h2>
                  <div className="weakness-ranking-list">
                    {currentWeakness.ranking.map((item, index) => <div className="weakness-ranking-item" key={item.topic}>
                        <span className="weakness-rank-number">{index + 1}</span>
                        <div className="weakness-rank-topic">
                          <strong>{item.topic}</strong>
                          <span>정답률 {item.accuracy}%</span>
                        </div>
                      </div>)}
                  </div>
                </section>
              </div>
              <section className="weakness-practice-card">
                <div className="weakness-practice-header">
                  <div>
                    <span className="weakness-page-label">FOCUSED PRACTICE</span>
                    <h2>취약 개념 집중 문제</h2>
                    <p>현재는 UI 시연용 샘플 문제이며, 추후 실제 학습 데이터와 연결할 예정입니다.</p>
                  </div>
                  <div className="weakness-practice-tags">
                    <span>{selectedCert}</span>
                    <span>{currentWeakness.focus}</span>
                  </div>
                </div>
                <div className="weakness-question-area">
                  <h3>Q. {currentWeakness.question}</h3>
                  <div className="weakness-options">
                    {currentWeakness.options.map((option, index) => {
                  const number = index + 1;
                  let optionClass = "weakness-option";
                  if (weaknessAnswer === number) optionClass += " selected";
                  if (weaknessSubmitted) {
                    if (number === currentWeakness.correctAnswer) {
                      optionClass += " correct";
                    } else if (number === weaknessAnswer && weaknessAnswer !== currentWeakness.correctAnswer) {
                      optionClass += " incorrect";
                    }
                  }
                  return <button type="button" key={number} className={optionClass} disabled={weaknessSubmitted} onClick={() => setWeaknessAnswer(number)}>
                          <span className="weakness-option-number">{number}</span>
                          <span>{option}</span>
                        </button>;
                })}
                  </div>
                  {!weaknessSubmitted ? <div className="weakness-submit-area">
                      <button type="button" className="weakness-submit-button" disabled={weaknessAnswer === null} onClick={submitWeaknessAnswer}>
                        정답 제출
                      </button>
                    </div> : <div className="weakness-result">
                      <div className={`weakness-result-title ${weaknessAnswer === currentWeakness.correctAnswer ? "success" : "wrong"}`}>
                        {weaknessAnswer === currentWeakness.correctAnswer ? "정답입니다!" : "오답입니다."}
                      </div>
                      <div className="weakness-explanation">
                        <h3>핵심 해설</h3>
                        <p>{currentWeakness.explanation}</p>
                      </div>
                      <button type="button" className="weakness-retry-button" onClick={resetWeaknessQuestion}>
                        다시 풀기
                      </button>
                    </div>}
                </div>
              </section>
            </div>
          </section>}
        {/* =================================================
            오답노트 / 취약점 기반 요약 노트
         ================================================= */}
        {currentPage === "wrongnote" && <section className="wrongnote-page">
            <div className="wrongnote-content">
              <div className="wrongnote-header">
                <div>
                  <span className="wrongnote-page-label">WRONG ANSWER NOTE</span>
                  <h1>나의 요약 노트</h1>
                  <p>
                    취약점 분석 결과를 바탕으로 시험 전에 다시 확인할 핵심 내용을
                    정리했습니다.
                  </p>
                </div>
                <div className="wrongnote-header-actions">
                  <span className="wrongnote-sample-badge">FRONTEND SAMPLE</span>
                  <button type="button" className="wrongnote-download-button" onClick={downloadSummaryNote}>
                    요약 노트 다운로드
                  </button>
                </div>
              </div>
              <section className="wrongnote-focus-section">
                <div className="wrongnote-section-heading">
                  <div>
                    <span className="wrongnote-section-kicker">WEAKNESS TOP 3</span>
                    <h2>집중 분석된 취약 영역</h2>
                  </div>
                  <span className="wrongnote-cert-chip">{selectedCert}</span>
                </div>
                <div className="wrongnote-top-grid">
                  {currentSummaryNotes.map(note => <div className="wrongnote-top-item" key={`top-${note.topic}`}>
                      <span className="wrongnote-top-rank">{note.rank}</span>
                      <div className="wrongnote-top-info">
                        <strong>{note.topic}</strong>
                        <span>현재 정답률 {note.accuracy}%</span>
                      </div>
                      <div className="wrongnote-top-score">{note.accuracy}%</div>
                    </div>)}
                </div>
              </section>
              <div className="wrongnote-note-grid">
                {currentSummaryNotes.map(note => <article className="wrongnote-card" key={note.topic}>
                    <div className="wrongnote-card-header">
                      <span className="wrongnote-card-number">
                        {String(note.rank).padStart(2, "0")}
                      </span>
                      <div>
                        <span className="wrongnote-card-label">WEAK TOPIC</span>
                        <h2>{note.topic}</h2>
                      </div>
                      <span className="wrongnote-card-accuracy">
                        {note.accuracy}%
                      </span>
                    </div>
                    <div className="wrongnote-card-section">
                      <h3>핵심 개념</h3>
                      <div className="wrongnote-concepts">
                        {note.concepts.map(concept => <span key={concept}>{concept}</span>)}
                      </div>
                    </div>
                    <div className="wrongnote-card-section">
                      <h3>자주 틀리는 포인트</h3>
                      <ul className="wrongnote-mistake-list">
                        {note.mistakes.map(mistake => <li key={mistake}>{mistake}</li>)}
                      </ul>
                    </div>
                    <div className="wrongnote-memory-box">
                      <span>시험 직전 암기</span>
                      <p>{note.memory}</p>
                    </div>
                  </article>)}
              </div>
              <div className="wrongnote-footer-guide">
                <span>NOTE</span>
                현재 화면은 프론트엔드 시연용 샘플입니다. 추후 실제 취약점 분석
                결과와 AI 생성 요약 내용을 API로 연결할 수 있습니다.
              </div>
            </div>
          </section>}
        {/* =================================================
            모의고사
         ================================================= */}
        {currentPage === "mockexam" && <section className="mock-page">
            <div className="mock-content">
              {mockStage === "setup" && <>
                  <div className="mock-page-header">
                    <div>
                      <span className="mock-page-label">MOCK EXAM</span>
                      <h1>실전 모의고사</h1>
                      <p>
                        응시 과목과 문제 수를 선택한 뒤 실제 시험처럼 문제를 풀어보세요.
                      </p>
                    </div>
                    <span className="mock-sample-badge">FRONTEND SAMPLE</span>
                  </div>
                  <div className="mock-setup-grid">
                    <section className="mock-setup-card mock-subject-card">
                      <div className="mock-card-heading">
                        <span>01</span>
                        <div>
                          <h2>응시 과목</h2>
                          <p>모의고사에 포함할 과목을 선택하세요.</p>
                        </div>
                      </div>
                      <div className="mock-subject-list">
                        {currentMockExam.subjects.map(subject => {
                    const active = mockSelectedSubjects.includes(subject);
                    return <button type="button" key={subject} className={`mock-subject-option ${active ? "selected" : ""}`} onClick={() => toggleMockSubject(subject)}>
                              <span className="mock-checkbox">
                                {active ? "✓" : ""}
                              </span>
                              <span>{subject}</span>
                            </button>;
                  })}
                      </div>
                    </section>
                    <section className="mock-setup-card">
                      <div className="mock-card-heading">
                        <span>02</span>
                        <div>
                          <h2>문제 수</h2>
                          <p>이번 모의고사에서 풀 문제 수를 선택하세요.</p>
                        </div>
                      </div>
                      <div className="mock-count-options">
                        {[10, 20, 30, 40, 50].map(count => <button type="button" key={count} className={mockQuestionCount === count ? "selected" : ""} onClick={() => setMockQuestionCount(count)}>
                            <strong>{count}</strong>
                            <span>문제</span>
                          </button>)}
                      </div>
                      <div className="mock-setup-summary">
                        <div>
                          <span>선택 자격증</span>
                          <strong>{selectedCert}</strong>
                        </div>
                        <div>
                          <span>선택 과목</span>
                          <strong>{mockSelectedSubjects.length}개</strong>
                        </div>
                        <div>
                          <span>총 문제</span>
                          <strong>{mockQuestionCount}문제</strong>
                        </div>
                      </div>
                      <button type="button" className="mock-start-button" onClick={startMockExam} disabled={mockSelectedSubjects.length === 0}>
                        모의고사 시작
                      </button>
                      <p className="mock-setup-caption">
                        현재는 UI 시연용 샘플 문제를 반복 구성합니다. 추후 실제 문제
                        API와 연결할 수 있습니다.
                      </p>
                    </section>
                  </div>
                </>}
              {mockStage === "exam" && mockCurrentQuestion && <>
                  <div className="mock-exam-header">
                    <div>
                      <span className="mock-page-label">IN PROGRESS</span>
                      <h1>실전 모의고사</h1>
                    </div>
                    <div className="mock-exam-progress-info">
                      <strong>
                        {mockIndex + 1} / {mockQuestions.length}
                      </strong>
                      <span>답변 완료 {mockAnsweredCount}문제</span>
                    </div>
                  </div>
                  <div className="mock-progress-track">
                    <div className="mock-progress-value" style={{
                width: `${(mockIndex + 1) / mockQuestions.length * 100}%`
              }} />
                  </div>
                  <div className="mock-exam-workspace">
                    <section className="mock-question-card">
                      <div className="mock-question-meta">
                        <span>{selectedCert}</span>
                        <span>{mockCurrentQuestion.subject}</span>
                      </div>
                      <div className="mock-question-number">
                        QUESTION {String(mockIndex + 1).padStart(2, "0")}
                      </div>
                      <h2>Q. {mockCurrentQuestion.question}</h2>
                      <div className="mock-answer-options">
                        {mockCurrentQuestion.options.map((option, index) => {
                    const answerNumber = index + 1;
                    const selected = mockAnswers[mockIndex] === answerNumber;
                    return <button type="button" key={answerNumber} className={`mock-answer-option ${selected ? "selected" : ""}`} onClick={() => selectMockAnswer(answerNumber)}>
                              <span>{answerNumber}</span>
                              <p>{option}</p>
                            </button>;
                  })}
                      </div>
                      <div className="mock-question-actions">
                        <button type="button" className="mock-secondary-button" onClick={() => moveMockQuestion(mockIndex - 1)} disabled={mockIndex === 0}>
                          이전 문제
                        </button>
                        {mockIndex < mockQuestions.length - 1 ? <button type="button" className="mock-primary-button" onClick={() => moveMockQuestion(mockIndex + 1)}>
                            다음 문제
                          </button> : <button type="button" className="mock-primary-button" onClick={openMockSubmit}>
                            시험 제출
                          </button>}
                      </div>
                    </section>
                    <aside className="mock-navigator-card">
                      <div className="mock-navigator-header">
                        <div>
                          <span className="mock-page-label">QUESTION MAP</span>
                          <h2>문제 현황</h2>
                        </div>
                        <span>{mockAnsweredCount}/{mockQuestions.length}</span>
                      </div>
                      <div className="mock-question-map">
                        {mockQuestions.map((question, index) => {
                    const answered = mockAnswers[index] !== undefined;
                    const current = index === mockIndex;
                    return <button type="button" key={question.instanceId} className={`${answered ? "answered" : ""} ${current ? "current" : ""}`} onClick={() => moveMockQuestion(index)}>
                              {index + 1}
                            </button>;
                  })}
                      </div>
                      <div className="mock-map-legend">
                        <span><i className="current" /> 현재 문제</span>
                        <span><i className="answered" /> 답변 완료</span>
                        <span><i /> 미응답</span>
                      </div>
                      <div className="mock-navigator-footer">
                        <span>정답은 시험 제출 후 공개됩니다.</span>
                        <button type="button" onClick={openMockSubmit}>
                          모의고사 제출
                        </button>
                      </div>
                    </aside>
                  </div>
                  {mockShowSubmitModal && <div className="mock-submit-overlay">
                      <div className="mock-submit-modal">
                        <span className="mock-page-label">SUBMIT EXAM</span>
                        <h2>모의고사를 제출할까요?</h2>
                        <p>
                          제출 후에는 점수와 과목별 결과, 문제 해설을 확인할 수 있습니다.
                        </p>
                        <div className="mock-submit-stats">
                          <div>
                            <span>답변 완료</span>
                            <strong>{mockAnsweredCount}</strong>
                          </div>
                          <div>
                            <span>미응답</span>
                            <strong>{mockQuestions.length - mockAnsweredCount}</strong>
                          </div>
                        </div>
                        {mockAnsweredCount < mockQuestions.length && <div className="mock-submit-warning">
                            아직 답하지 않은 문제가 있습니다. 미응답 문제는 오답으로
                            처리됩니다.
                          </div>}
                        <div className="mock-submit-actions">
                          <button type="button" className="mock-secondary-button" onClick={() => setMockShowSubmitModal(false)}>
                            계속 풀기
                          </button>
                          <button type="button" className="mock-primary-button" onClick={submitMockExam}>
                            제출하기
                          </button>
                        </div>
                      </div>
                    </div>}
                </>}
              {mockStage === "result" && <>
                  <div className="mock-page-header mock-result-header">
                    <div>
                      <span className="mock-page-label">MOCK EXAM RESULT</span>
                      <h1>모의고사 결과</h1>
                      <p>과목별 성적과 틀린 문제를 확인하고 다음 학습 방향을 정리해보세요.</p>
                    </div>
                    <div className="mock-result-actions">
                      <button type="button" className="mock-secondary-button" onClick={restartMockExam}>
                        다시 응시
                      </button>
                      <button type="button" className="mock-primary-button" onClick={goToWeaknessFromMock}>
                        취약점 학습
                      </button>
                    </div>
                  </div>
                  <div className="mock-result-grid">
                    <section className="mock-score-card">
                      <span className="mock-page-label">TOTAL SCORE</span>
                      <div className="mock-score-circle">
                        <strong>{mockScore}</strong>
                        <span>점</span>
                      </div>
                      <h2>
                        {mockCorrectCount} / {mockQuestions.length}문제 정답
                      </h2>
                      <p>정답률 {mockScore}%</p>
                      <div className="mock-score-mini-stats">
                        <div>
                          <span>정답</span>
                          <strong>{mockCorrectCount}</strong>
                        </div>
                        <div>
                          <span>오답</span>
                          <strong>{mockQuestions.length - mockCorrectCount}</strong>
                        </div>
                        <div>
                          <span>미응답</span>
                          <strong>{mockQuestions.length - mockAnsweredCount}</strong>
                        </div>
                      </div>
                    </section>
                    <section className="mock-subject-result-card">
                      <div className="mock-result-section-heading">
                        <div>
                          <span className="mock-page-label">SUBJECT RESULT</span>
                          <h2>과목별 결과</h2>
                        </div>
                        <span>{selectedCert}</span>
                      </div>
                      <div className="mock-subject-results">
                        {mockSubjectStats.map(stat => <div className="mock-subject-result-row" key={stat.subject}>
                            <div className="mock-subject-result-text">
                              <strong>{stat.subject}</strong>
                              <span>
                                {stat.correct}/{stat.total} 정답 · {stat.accuracy}%
                              </span>
                            </div>
                            <div className="mock-result-bar">
                              <div style={{
                        width: `${stat.accuracy}%`
                      }} />
                            </div>
                          </div>)}
                      </div>
                    </section>
                  </div>
                  <section className="mock-review-section">
                    <div className="mock-review-header">
                      <div>
                        <span className="mock-page-label">REVIEW</span>
                        <h2>문제 복기</h2>
                        <p>문제 번호를 선택하면 내 답과 정답, 해설을 확인할 수 있습니다.</p>
                      </div>
                      <div className="mock-review-legend">
                        <span><i className="correct" /> 정답</span>
                        <span><i className="wrong" /> 오답</span>
                      </div>
                    </div>
                    <div className="mock-review-layout">
                      <div className="mock-review-map">
                        {mockQuestions.map((question, index) => {
                    const isCorrect = mockAnswers[index] === question.answer;
                    return <button type="button" key={`review-${question.instanceId}`} className={`${isCorrect ? "correct" : "wrong"} ${mockReviewIndex === index ? "selected" : ""}`} onClick={() => setMockReviewIndex(index)}>
                              <span>{index + 1}</span>
                              <small>{isCorrect ? "O" : "X"}</small>
                            </button>;
                  })}
                      </div>
                      {mockReviewQuestion && <article className="mock-review-detail">
                          <div className="mock-review-detail-top">
                            <span>{mockReviewQuestion.subject}</span>
                            <strong>
                              {mockAnswers[mockReviewIndex] === mockReviewQuestion.answer ? "정답" : "오답"}
                            </strong>
                          </div>
                          <h3>
                            Q{mockReviewIndex + 1}. {mockReviewQuestion.question}
                          </h3>
                          <div className="mock-review-answer-row">
                            <div>
                              <span>내 답</span>
                              <strong>
                                {mockAnswers[mockReviewIndex] ? `${mockAnswers[mockReviewIndex]}번` : "미응답"}
                              </strong>
                            </div>
                            <div>
                              <span>정답</span>
                              <strong>{mockReviewQuestion.answer}번</strong>
                            </div>
                          </div>
                          <div className="mock-review-explanation">
                            <span>핵심 해설</span>
                            <p>{mockReviewQuestion.explanation}</p>
                          </div>
                        </article>}
                    </div>
                  </section>
                </>}
            </div>
          </section>}
        {/* =================================================
            사용자 / 학습 현황
         ================================================= */}
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
                <button type="button" className="learning-reset-button" onClick={() => alert("학습 데이터 초기화 기능은 추후 연결 예정입니다.")}>
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
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={currentDashboard.radar} outerRadius="68%">
                      <PolarGrid stroke={darkMode ? "#464646" : "#d4d4d4"} />
                      <PolarAngleAxis dataKey="subject" tick={{
                    fill: darkMode ? "#c6c6c6" : "#444444",
                    fontSize: 12
                  }} />
                      <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{
                    fill: darkMode ? "#777777" : "#888888",
                    fontSize: 10
                  }} />
                      <Radar name="정답률" dataKey="score" stroke="#4da3ff" fill="#4da3ff" fillOpacity={0.32} strokeWidth={2} />
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
                      {currentDashboard.weakest}
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
                      {currentDashboard.todaySolved}
                      <span>
                        {" "}/{" "}
                        {currentDashboard.dailyGoal}
                      </span>
                    </div>
                    <div className="goal-progress">
                      <div className="goal-progress-bar" style={{
                    width: `${goalRate}%`
                  }} />
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
                        {currentDashboard.totalSolved}
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
                        {currentDashboard.correctSolved}
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
                        {currentDashboard.accuracy}
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
                  <button type="button" className={`cert-card ${selectedCert === "정보처리기사" ? "selected" : ""}`} onClick={() => setSelectedCert("정보처리기사")}>
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
                  <button type="button" className={`cert-card ${selectedCert === "공인중개사 1차" ? "selected" : ""}`} onClick={() => setSelectedCert("공인중개사 1차")}>
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
