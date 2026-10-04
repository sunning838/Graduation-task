import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext.jsx';
import LearningPage from './LearningPage.jsx';

export default function HomePage() {
  const { user } = useAuth();

  // 로그인하면 학습 화면을 보여줌
  if (user) return <LearningPage />;

  return (
    <section className="home">
      <h1 className="home-title">개념을 이해하고, 문제로 확인하세요</h1>
      <p className="home-desc">
        자격증 단원별 개념 강의를 AI 튜터와 함께 듣고, 궁금한 부분은 바로 질문할 수 있어요.
      </p>
      <div className="home-actions">
        <Link to="/signup" className="btn btn-primary btn-lg">회원가입하고 시작하기</Link>
        <Link to="/login" className="btn btn-ghost btn-lg">이미 계정이 있어요</Link>
      </div>
    </section>
  );
}