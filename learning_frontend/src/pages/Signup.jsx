import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AuthSide from "./AuthSide";
import "./Auth.css";

export default function Signup() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
    passwordConfirm: "",
  });
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setError("");
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!form.name || !form.email || !form.password || !form.passwordConfirm) {
      setError("모든 항목을 입력해주세요.");
      return;
    }
    if (form.password.length < 8) {
      setError("비밀번호는 8자 이상으로 입력해주세요.");
      return;
    }
    if (form.password !== form.passwordConfirm) {
      setError("비밀번호가 일치하지 않습니다. 다시 확인해주세요.");
      return;
    }
    // TODO: 백엔드 연결 전까지는 저장 없이 로그인 화면으로 이동
    navigate("/login");
  };

  return (
    <div className="auth-page">
      <AuthSide />
      <main className="auth-main">
        <div className="auth-card">
          <h1>회원가입</h1>
          <p className="auth-sub">계정을 만들고 AI 튜터와 학습을 시작하세요.</p>

          {error && <div className="auth-error">{error}</div>}

          <form className="auth-form" onSubmit={handleSubmit}>
            <div className="auth-field">
              <label htmlFor="name">이름</label>
              <input
                id="name"
                name="name"
                type="text"
                placeholder="홍길동"
                value={form.name}
                onChange={handleChange}
                autoComplete="name"
              />
            </div>
            <div className="auth-field">
              <label htmlFor="email">이메일</label>
              <input
                id="email"
                name="email"
                type="email"
                placeholder="example@anyang.ac.kr"
                value={form.email}
                onChange={handleChange}
                autoComplete="email"
              />
            </div>
            <div className="auth-field">
              <label htmlFor="password">비밀번호</label>
              <input
                id="password"
                name="password"
                type="password"
                placeholder="8자 이상"
                value={form.password}
                onChange={handleChange}
                autoComplete="new-password"
              />
            </div>
            <div className="auth-field">
              <label htmlFor="passwordConfirm">비밀번호 확인</label>
              <input
                id="passwordConfirm"
                name="passwordConfirm"
                type="password"
                placeholder="비밀번호 다시 입력"
                value={form.passwordConfirm}
                onChange={handleChange}
                autoComplete="new-password"
              />
            </div>
            <button type="submit" className="auth-submit">회원가입</button>
          </form>

          <p className="auth-switch">
            이미 계정이 있나요?<Link to="/login">로그인</Link>
          </p>
        </div>
      </main>
    </div>
  );
}
