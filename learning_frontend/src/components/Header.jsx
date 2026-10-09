import { Link } from 'react-router-dom';
import ThemeToggle from './ThemeToggle.jsx';

export default function Header() {
  return (
    <header className="header">
      <div className="header-inner">
        {/* 제목을 누르면 홈 화면으로 이동 */}
        <Link to="/" className="brand">자격증 AI 학습실</Link>

        <nav className="header-actions">
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}