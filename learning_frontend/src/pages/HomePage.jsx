import { Link } from 'react-router-dom';
import './home.css';

const FEATURES = [
  {
    title: '세 가지 설명 방식',
    desc: '기본 설명이 어렵다면 쉬운 말로, 그래도 막히면 예시로 다시 봅니다.',
  },
  {
    title: 'AI 튜터에게 바로 질문',
    desc: '보고 있는 강의를 기준으로 답해 줘서, 이해 안 되는 부분만 콕 집어 물어볼 수 있어요.',
  },
  {
    title: '음성으로 듣는 강의',
    desc: '강의를 소리로 들으며 이동 중에도 귀로 복습하세요.',
  },
  {
    title: '단원별 진도 관리',
    desc: '과목마다 어디까지 공부했는지 한눈에 보고, 이어서 공부할 수 있어요.',
  },
];

const STEPS = [
  { title: '단원 고르기', desc: '처음부터, 이어서, 또는 원하는 단원부터 시작하세요.' },
  { title: '강의 보고 듣기', desc: '내 수준에 맞는 설명 방식을 골라 읽거나 들어요.' },
  { title: '묻고, 문제로 확인하기', desc: '막히면 튜터에게 묻고, 문제풀이로 실력을 점검해요.' },
];

export default function HomePage() {
  return (
    <div className="hp">
      {/* 1. 첫 화면 */}
      <section className="hp-hero">
        <h1 className="hp-title">강의 듣고, 문제 풀고, 모르면 바로 물어보는 AI 튜터</h1>
        <p className="hp-desc">개념부터 문제풀이까지, AI 튜터에게 대화로 물어보세요.</p>
        <div className="hp-actions">
          <Link to="/learn" className="btn btn-primary btn-lg">학습 시작하기</Link>
        </div>
      </section>

      {/* 2. 주요 기능 */}
      <section className="hp-section">
        <div className="hp-section-head">
          <h2>혼자 공부해도 막히지 않게</h2>
          <p>이해될 때까지 다시 설명하고, 궁금한 건 바로 묻는 학습실</p>
        </div>
        <div className="hp-grid">
          {FEATURES.map((f, i) => (
            <div key={f.title} className="hp-card">
              <span className="hp-card-num">{String(i + 1).padStart(2, '0')}</span>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* 3. 이용 방법 */}
      <section className="hp-section" style={{ paddingBottom: '4rem' }}>
        <div className="hp-section-head">
          <h2>이렇게 공부해요</h2>
        </div>
        <div className="hp-steps">
          {STEPS.map((s, i) => (
            <div key={s.title} className="hp-card">
              <span className="hp-step-num">{i + 1}</span>
              <h3>{s.title}</h3>
              <p>{s.desc}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}