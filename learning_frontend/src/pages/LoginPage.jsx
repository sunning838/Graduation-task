import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext.jsx';

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!email.trim() || !password) {
      setError('이메일과 비밀번호를 입력하세요.');
      return;
    }
    setError('');
    setLoading(true);
    try {
      await login(email.trim(), password);
      navigate('/');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const onKeyDown = (e) => {
    if (e.key === 'Enter') handleLogin();
  };

  return (
    <section className="auth">
      <div className="auth-card">
        <h1 className="auth-title">로그인</h1>
        <p className="auth-sub">이어서 공부하려면 로그인하세요.</p>

        <label className="field">
          <span className="field-label">이메일</span>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="example@email.com"
            autoComplete="email"
          />
        </label>

        <label className="field">
          <span className="field-label">비밀번호</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={onKeyDown}
            autoComplete="current-password"
          />
        </label>

        {error && <p className="form-error" role="alert">{error}</p>}

        <button type="button" className="btn btn-primary btn-block" onClick={handleLogin} disabled={loading}>
          {loading ? '로그인하는 중…' : '로그인'}
        </button>

        <p className="auth-switch">
          계정이 없나요? <Link to="/signup">회원가입</Link>
        </p>
      </div>
    </section>
  );
}
