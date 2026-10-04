# 자격증 AI 학습실 – React 프론트엔드

## 실행 방법
```
npm install
npm run dev
```
브라우저에서 http://localhost:5173 접속

## 현재 구현된 것
- 상단 헤더 (제목, 라이트/다크 모드 전환, 로그인·회원가입 버튼)
- 로그인 / 회원가입 화면 (버튼 클릭 시 페이지 이동)
- 테마는 브라우저에 저장되어 새로고침해도 유지

## 백엔드 연결
`src/api/api.js` 한 파일만 수정하면 됨.
지금은 `USE_MOCK = true`라 브라우저(localStorage)에 가짜 회원 정보를 저장함.
백엔드가 준비되면 `USE_MOCK = false`로 바꾸고 `BASE_URL`, 경로(`/auth/login`, `/auth/signup`)를 맞추면 됨.

## 폴더 구조
```
src/
  api/api.js              백엔드 호출 (현재 mock)
  context/ThemeContext    라이트/다크 모드 상태
  context/AuthContext     로그인 상태
  components/Header       상단 바
  components/ThemeToggle  테마 전환 버튼
  pages/HomePage          메인 (학습 화면이 들어갈 자리)
  pages/LoginPage         로그인
  pages/SignupPage        회원가입
  styles/global.css       색상 토큰 + 전체 스타일
```
