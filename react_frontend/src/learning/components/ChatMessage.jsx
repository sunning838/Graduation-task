import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import GraphDiagram from './GraphDiagram.jsx';

export default function ChatMessage({ message }) {
  const { role, text, visual, research } = message;

  return (
    <div className={`msg msg-${role}`}>
      <div className="msg-role">{role === 'user' ? '나' : '튜터'}</div>
      <div className="markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
      </div>
      {visual && <Visual visual={visual} />}
      {research?.status === 'supplemented' && <Research research={research} />}
    </div>
  );
}

export function Visual({ visual }) {
  if (visual.kind !== 'table' && visual.kind !== 'graph') return null;
  return (
    <div className="visual">
      <h4>{visual.title}</h4>
      <p className="muted small">{visual.purpose}</p>
      {visual.kind === 'table' ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {visual.columns.map((c) => (
                  <th key={c}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {visual.rows.map((row, i) => (
                <tr key={i}>
                  {row.map((cell, j) => (
                    <td key={j}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <GraphDiagram visual={visual} />
      )}
    </div>
  );
}

export function Research({ research }) {
  return (
    <div className="research">
      <p className="muted small">웹 검색 참고</p>
      {research.suggestions_html && (
        <iframe title="웹 검색 참고" srcDoc={research.suggestions_html} sandbox="" />
      )}
      <div className="research-links">
        {(research.sources ?? []).map((s) => (
          <a key={s.url} className="btn btn-ghost" href={s.url} target="_blank" rel="noopener noreferrer">
            {s.title}
          </a>
        ))}
      </div>
    </div>
  );
}
