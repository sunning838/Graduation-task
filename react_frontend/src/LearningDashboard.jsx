import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts';

export function StatsStatus({ loading, error, onRetry }) {
  if (loading) return <p role="status" className="learning-data-status">학습 기록을 불러오고 있습니다.</p>;
  if (error) return <div role="alert" className="learning-data-status">{error} <button type="button" onClick={onRetry}>다시 조회</button></div>;
  return null;
}

export default function LearningDashboard({ stats, loading, error, onRetry, darkMode }) {
  if (!stats) return <StatsStatus loading={loading} error={error} onRetry={onRetry} />;
  const learned = stats.subjects.filter(subject => subject.total > 0);
  return <>
    <section className="dashboard-card weakness-card">
      <div className="dashboard-card-header">
        <div><span className="dashboard-small-label">LEARNING ANALYSIS</span><h2>나의 학습 취약점 분석</h2></div>
        <button type="button" onClick={onRetry}>새로고침</button>
      </div>
      {!stats.has_records && <p className="learning-data-status">아직 풀이 기록이 없습니다. 일반 문제풀이부터 시작해 보세요.</p>}
      {learned.length >= 3 && <>
        <p>풀이 기록이 있는 과목의 정답률입니다.</p>
        <div className="radar-chart-wrapper"><ResponsiveContainer width="100%" height="100%">
          <RadarChart data={learned} outerRadius="68%">
            <PolarGrid stroke={darkMode ? '#464646' : '#d4d4d4'} />
            <PolarAngleAxis dataKey="label" tick={{ fill: darkMode ? '#c6c6c6' : '#444', fontSize: 12 }} />
            <PolarRadiusAxis angle={90} domain={[0, 100]} />
            <Radar name="정답률" dataKey="accuracy" stroke="#4da3ff" fill="#4da3ff" fillOpacity={0.32} />
          </RadarChart>
        </ResponsiveContainer></div>
      </>}
      <div className="learning-subject-list">
        {stats.subjects.map(subject => <div key={subject.topic} className="learning-subject-row">
          <strong>{subject.label}</strong>
          <span>{subject.total ? `${subject.correct}/${subject.total}문제 정답 · ${subject.accuracy}%` : '미학습'}</span>
          {subject.status === 'insufficient_data' && <small>분석할 기록 부족</small>}
        </div>)}
      </div>
      <div className="weakness-analysis">{stats.weakness.message}</div>
      <p className="summary-caption">과목별 {stats.weakness.min_attempts}문제 이상, 정답률 {stats.weakness.accuracy_threshold}% 미만을 취약 과목으로 분류합니다.</p>
      {stats.unclassified_solved > 0 && <p>과목이 확인되지 않은 기존 기록 {stats.unclassified_solved}건은 전체 통계에만 포함됩니다.</p>}
    </section>
    <div className="learning-summary-grid">
      <section className="dashboard-card summary-card">
        <div className="summary-title"><h2>🎯 오늘의 목표 달성률</h2></div>
        <div className="summary-content">
          <span className="summary-label">오늘 풀이 수 · 한국 시간 기준</span>
          <div className="summary-big-number">{stats.today_solved}<span> / {stats.daily_goal}</span></div>
          <div className="goal-progress"><div className="goal-progress-bar" style={{ width: `${stats.goal_rate}%` }} /></div>
          <div className="summary-caption">현재 목표 달성률: {stats.goal_rate}%</div>
        </div>
      </section>
      <section className="dashboard-card summary-card">
        <div className="summary-title"><h2>📊 전체 학습 현황</h2></div>
        <div className="total-stats">
          <div className="total-stat-item"><span>푼 문제</span><strong>{stats.total_solved}<small>개</small></strong></div>
          <div className="total-stat-item"><span>맞춘 문제</span><strong>{stats.correct_solved}<small>개</small></strong></div>
          <div className="total-stat-item"><span>정답률</span><strong>{stats.accuracy === null ? '—' : `${stats.accuracy}%`}</strong></div>
        </div>
      </section>
    </div>
  </>;
}
