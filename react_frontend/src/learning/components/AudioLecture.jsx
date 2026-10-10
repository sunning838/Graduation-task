import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import * as api from '../api/api.js';
import ChatMessage, { Visual, Research } from './ChatMessage.jsx';

const SPEEDS = [0.75, 1, 1.25, 1.5, 2];
const speedLabel = (s) => `${Number.isInteger(s * 2) ? s.toFixed(1) : s}x`;

export default function AudioLecture({ lessonId, variant, message }) {
  const [pkg, setPkg] = useState(null); // { audio_url, timeline }
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [active, setActive] = useState(-1);
  const [ended, setEnded] = useState(false);
  const [follow, setFollow] = useState(true);
  const [speed, setSpeed] = useState(1);

  const audioRef = useRef(null);
  const boxRef = useRef(null);
  const segRefs = useRef([]);

  const generate = async () => {
    setLoading(true);
    setError('');
    try {
      setPkg(await api.createAudioLecture(lessonId, variant, message.text));
    } catch (e) {
      setError(e.message || '음성 강의를 만들지 못했습니다. 다시 시도해 주세요.');
    } finally {
      setLoading(false);
    }
  };

  // 재생 속도 적용
  useEffect(() => {
    if (audioRef.current) audioRef.current.playbackRate = speed;
  }, [speed, pkg]);

  // 읽는 문단을 가운데로 자동 스크롤
  useEffect(() => {
    if (!follow || active < 0) return;
    const box = boxRef.current;
    const el = segRefs.current[active];
    if (!box || !el) return;
    box.scrollTo({
      top: Math.max(0, el.offsetTop - box.clientHeight / 2 + el.offsetHeight / 2),
      behavior: 'smooth',
    });
  }, [active, follow]);

  // 지금 재생 시간에 해당하는 문단 찾기
  const sync = () => {
    const audio = audioRef.current;
    if (!audio || !pkg) return;
    const t = audio.currentTime || 0;
    let index = 0;
    for (let i = pkg.timeline.length - 1; i >= 0; i -= 1) {
      if (t >= pkg.timeline[i].start) {
        index = i;
        break;
      }
    }
    setActive(index);
    setEnded(false);
  };

  // 문단을 누르면 그 위치부터 재생
  const seek = (i) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = pkg.timeline[i].start;
    setActive(i);
    audio.play().catch(() => {});
  };

  const header = (
    <>
      <h4 className="audio-title">AI 음성 강의</h4>
      <p className="audio-desc">음성을 재생하면 현재 읽고 있는 강의 내용이 자동으로 강조됩니다.</p>
    </>
  );

  // 음성 만들기 전: 버튼 + 일반 강의 글
  if (!pkg) {
    return (
      <div>
        {header}
        <div className="audio-start">
          <button type="button" className="btn btn-ghost" onClick={generate} disabled={loading}>
            {loading ? '음성 강의를 만들고 있어요… 처음에는 1~2분 걸릴 수 있어요' : 'AI 음성 강의로 듣기'}
          </button>
          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
        </div>
        <ChatMessage message={message} />
      </div>
    );
  }

  const total = pkg.timeline.length;
  const status = ended
    ? '강의 재생이 끝났습니다.'
    : active >= 0
      ? `현재 강의 위치: ${active + 1} / ${total}`
      : '재생하면 현재 읽는 문단이 강조됩니다.';

  return (
    <div className="audio-lecture">
      <div className="audio-top">
        {header}
        <audio
          ref={audioRef}
          controls
          preload="metadata"
          src={api.audioUrl(pkg.audio_url)}
          onTimeUpdate={sync}
          onSeeking={sync}
          onLoadedMetadata={sync}
          onPlay={(e) => {
            e.currentTarget.playbackRate = speed;
          }}
          onEnded={() => {
            setActive(total - 1);
            setEnded(true);
          }}
        />
        <div className="audio-row">
          <label className="audio-follow">
            <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
            현재 읽는 부분 자동 따라가기
          </label>
          <label className="audio-speed">
            재생 속도
            <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
              {SPEEDS.map((s) => (
                <option key={s} value={s}>
                  {speedLabel(s)}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="audio-status">{status}</p>
      </div>

      <div className="audio-segments" ref={boxRef}>
        {pkg.timeline.map((seg, i) => (
          <section
            key={seg.index}
            ref={(el) => {
              segRefs.current[i] = el;
            }}
            className={`audio-segment${i === active ? ' is-active' : ''}`}
            onClick={() => seek(i)}
          >
            <div className="markdown">
              <ReactMarkdown>{seg.markdown}</ReactMarkdown>
            </div>
          </section>
        ))}
        {message.visual && <Visual visual={message.visual} />}
        {message.research?.status === 'supplemented' && <Research research={message.research} />}
      </div>
    </div>
  );
}