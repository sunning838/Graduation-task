import { Routes, Route, Navigate } from 'react-router-dom';
import Header from './components/Header.jsx';
import HomePage from './pages/HomePage.jsx';
import LearningPage from './pages/LearningPage.jsx';

export default function App() {
  return (
    <div className="app">
      <Header />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/learn" element={<LearningPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}