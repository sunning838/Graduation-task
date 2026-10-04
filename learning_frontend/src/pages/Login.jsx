import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AuthSide from "./AuthSide";
import "./Auth.css";

export default function Login() {
  const navigate = useNavigate();
  const [form, setForm] = useState({ email: "", password: "" });
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setError("");
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!form.email || !form.password) {
      setError("이메일과 비밀번호를 모두 입력해주세요.");
      return;
    }
    // TODO: 백엔드 연결 전까지는 검증 없이 메인 화면으로 이동
    navigate("/");
  };

  return (
    <div className="auth-page">
      <AuthSide />
      <main className="auth-main">
        <div className="auth-card">
          <h1>로그인</h1>
          <p className="auth-sub">계정에 로그인하고 학습을 이어가세요.</p>

          {error && <div className="auth-error">{error}</div>}

          <form className="auth-form" onSubmit={handleSubmit}>
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
                placeholder="비밀번호 입력"
                value={form.password}
                onChange={handleChange}
                autoComplete="current-password"
              />
            </div>
            <button type="submit" className="auth-submit">로그인</button>
          </form>

          <p className="auth-switch">
            계정이 없나요?<Link to="/signup">회원가입</Link>
          </p>
        </div>
      </main>
    </div>
  );
}
