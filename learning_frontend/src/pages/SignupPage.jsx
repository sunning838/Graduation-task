import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext.jsx';

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export default function SignupPage() {
  const { signup } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({ name: '', email: '', password: '', confirm: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const update = (key) => (e) => setForm({ ...form, [key]: e.target.value });

  const validate = () => {
    if (!form.name.trim()) return '이름을 입력하세요.';
    if (!EMAIL_RE.test(form.email.trim())) return '이메일 형식을 확인하세요.';
    if (form.password.length < 8) return '비밀번호는 8자 이상으로 입력하세요.';
    if (form.password !== form.confirm) return '비밀번호 확인이 일치하지 않습니다.';
    return '';
  };

  const handleSignup = async () => {
    const message = validate();
    if (message) {
      setError(message);
      return;
    }
    setError('');
    setLoading(true);
    try {
      await signup(form.name.trim(), form.email.trim(), form.password);
      navigate('/learn'); // 가입하면 학습 화면으로
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.nativeEvent.isComposing) handleSignup();
  };

  return (
    <section className="auth">
      <div className="auth-card">
        <h1 className="auth-title">회원가입</h1>
        <p className="auth-sub">가입하면 학습 진도가 계정에 저장돼요.</p>

        <label className="field">
          <span className="field-label">이름</span>
          <input value={form.name} onChange={update('name')} onKeyDown={onKeyDown} autoComplete="name" />
        </label>

        <label className="field">
          <span className="field-label">이메일</span>
          <input
            type="email"
            value={form.email}
            onChange={update('email')}
            onKeyDown={onKeyDown}
            placeholder="example@email.com"
            autoComplete="email"
          />
        </label>

        <label className="field">
          <span className="field-label">비밀번호</span>
          <input
            type="password"
            value={form.password}
            onChange={update('password')}
            onKeyDown={onKeyDown}
            autoComplete="new-password"
          />
          <span className="field-hint">8자 이상</span>
        </label>

        <label className="field">
          <span className="field-label">비밀번호 확인</span>
          <input
            type="password"
            value={form.confirm}
            onChange={update('confirm')}
            onKeyDown={onKeyDown}
            autoComplete="new-password"
          />
        </label>

        {error && <p className="form-error" role="alert">{error}</p>}

        <button type="button" className="btn btn-primary btn-block" onClick={handleSignup} disabled={loading}>
          {loading ? '가입하는 중…' : '회원가입'}
        </button>

        <p className="auth-switch">
          이미 계정이 있나요? <Link to="/login">로그인</Link>
        </p>
      </div>
    </section>
  );
}