import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './context/AuthContext.jsx';
import Header from './components/Header.jsx';
import HomePage from './pages/HomePage.jsx';
import LearningPage from './pages/LearningPage.jsx';
import LoginPage from './pages/LoginPage.jsx';
import SignupPage from './pages/SignupPage.jsx';

export default function App() {
  const { user } = useAuth();

  return (
    <div className="app">
      <Header />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<HomePage />} />
          {/* 학습 화면은 로그인해야 볼 수 있음 */}
          <Route path="/learn" element={user ? <LearningPage /> : <Navigate to="/login" replace />} />
          <Route path="/login" element={user ? <Navigate to="/learn" replace /> : <LoginPage />} />
          <Route path="/signup" element={user ? <Navigate to="/learn" replace /> : <SignupPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}