import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import * as api from '../api/lessonClient.js';
import { Visual, Research } from './ChatMessage.jsx';
import { localLecture, lectureFromSegments, slideForSegment, timelineSegment, timeLabel } from './lectureSlides.js';

const COMPLETE = new Set(['ready', 'partial', 'failed']);

export default function PresentationLecture({ lessonId, variant, message, onOpenTutor }) {
  const local = useMemo(() => localLecture(message.text), [message.text]);
  const [presentation, setPresentation] = useState(null);
  const [imageError, setImageError] = useState('');
  const [imageReload, setImageReload] = useState(0);
  const [audio, setAudio] = useState(null);
  const [audioLoading, setAudioLoading] = useState(false);
  const [audioError, setAudioError] = useState('');
  const [playing, setPlaying] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [duration, setDuration] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [slideIndex, setSlideIndex] = useState(0);
  const [activeSegment, setActiveSegment] = useState(-1);
  const [follow, setFollow] = useState(true);
  const [failedImages, setFailedImages] = useState({});
  const audioRef = useRef(null);
  const transcriptRef = useRef(null);
  const paragraphRefs = useRef([]);
  const playRequested = useRef(false);
  const requestedSegment = useRef(0);
  const audioController = useRef(null);
  const imageRetryController = useRef(null);
  const expandedRef = useRef(null);
  const fallback = useMemo(() => audio?.timeline ? lectureFromSegments(audio.timeline) : local, [audio, local]);
  const slides = presentation?.slides?.length ? presentation.slides : fallback.slides;
  const segments = audio?.timeline || presentation?.segments || local.segments;
  const current = slides[Math.min(slideIndex, slides.length - 1)];
  const imageBusy = !presentation || !COMPLETE.has(presentation.status);

  useEffect(() => {
    const controller = new AbortController();
    let timer;
    let failures = 0;
    const poll = async id => {
      try {
        const result = await api.getPresentation(id, controller.signal);
        if (controller.signal.aborted) return;
        setPresentation(result); setImageError(''); failures = 0;
        if (!COMPLETE.has(result.status)) timer = setTimeout(() => poll(id), 1800);
      } catch (error) {
        if (controller.signal.aborted) return;
        if (++failures < 3) timer = setTimeout(() => poll(id), 2500);
        else setImageError(error.message);
      }
    };
    api.startPresentation(lessonId, variant, message.text, controller.signal).then(result => {
      if (controller.signal.aborted) return;
      setPresentation(result); setImageError('');
      if (!COMPLETE.has(result.status)) timer = setTimeout(() => poll(result.package_id), 1500);
    }).catch(error => { if (!controller.signal.aborted) setImageError(error.message); });
    return () => { controller.abort(); clearTimeout(timer); };
  }, [lessonId, variant, message.text, imageReload]);

  useEffect(() => () => {
    audioController.current?.abort();
    imageRetryController.current?.abort();
  }, []);

  useEffect(() => {
    if (audioRef.current) audioRef.current.playbackRate = speed;
  }, [speed, audio]);

  useEffect(() => {
    if (!audio || !playRequested.current) return;
    const player = audioRef.current;
    const begin = () => {
      playRequested.current = false;
      const start = audio.timeline.find(segment => segment.index === requestedSegment.current);
      if (start) player.currentTime = start.start;
      player.play().catch(() => setAudioError('재생 버튼을 눌러 강의를 시작해 주세요.'));
    };
    if (player.readyState >= 1) begin();
    else player.addEventListener('loadedmetadata', begin, { once: true });
    return () => player.removeEventListener('loadedmetadata', begin);
  }, [audio]);

  const goToParagraph = index => {
    const box = transcriptRef.current, paragraph = paragraphRefs.current[index];
    if (!box || !paragraph) return;
    box.scrollTo({ top: Math.max(0, paragraph.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop - 55), behavior: 'smooth' });
  };

  useEffect(() => {
    if (!follow || activeSegment < 0) return;
    const box = transcriptRef.current, paragraph = paragraphRefs.current[activeSegment];
    if (box && paragraph) box.scrollTo({ top: Math.max(0, paragraph.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop - 55), behavior: 'smooth' });
  }, [activeSegment, follow]);

  const syncAudio = () => {
    const player = audioRef.current;
    if (!audio || !player) return;
    const time = player.currentTime || 0;
    const index = timelineSegment(audio.timeline, time);
    setSeconds(time); setActiveSegment(index);
    const page = slideForSegment(slides, audio.timeline[index]?.index ?? index);
    if (page >= 0) setSlideIndex(page);
  };

  const startAudio = async () => {
    if (audioLoading) return;
    setAudioError('');
    if (audio) {
      if (playing) audioRef.current.pause();
      else audioRef.current.play().catch(() => setAudioError('음성을 재생하지 못했습니다. 다시 눌러 주세요.'));
      return;
    }
    const controller = new AbortController(); audioController.current = controller;
    requestedSegment.current = current?.segment_indices[0] || 0;
    setAudioLoading(true); playRequested.current = true;
    try {
      const result = await api.createLectureAudio(lessonId, variant, message.text, controller.signal);
      if (controller.signal.aborted) return;
      if (!result.timeline?.length || !api.lectureMediaUrl(result.audio_url)) throw new Error('음성을 준비하지 못했습니다.');
      setAudio(result);
    } catch (error) {
      if (!controller.signal.aborted) { playRequested.current = false; setAudioError(error.message); }
    } finally { if (!controller.signal.aborted) setAudioLoading(false); }
  };

  const changeSlide = index => {
    if (index < 0 || index >= slides.length) return;
    setSlideIndex(index);
    const target = slides[index].segment_indices[0];
    if (audio && audioRef.current) {
      const item = audio.timeline.find(seg => seg.index === target);
      if (item) { audioRef.current.currentTime = item.start; setSeconds(item.start); setActiveSegment(target); }
    }
    if (follow) goToParagraph(target);
  };

  const seekParagraph = index => {
    if (!audioRef.current || !audio) return;
    audioRef.current.currentTime = audio.timeline[index].start;
    syncAudio();
    audioRef.current.play().catch(() => setAudioError('재생 버튼을 눌러 주세요.'));
  };

  const retryImages = async () => {
    setImageError('');
    setFailedImages({});
    if (!presentation) { setImageReload(x => x + 1); return; }
    imageRetryController.current?.abort();
    const controller = new AbortController(); imageRetryController.current = controller;
    try {
      const result = await api.retryPresentation(presentation.package_id, controller.signal);
      if (!controller.signal.aborted) { setPresentation(result); setImageReload(x => x + 1); }
    } catch (error) { if (!controller.signal.aborted) setImageError(error.message); }
  };

  const imageReady = current?.image_status === 'ready' && !failedImages[current.id];
  const imageUrl = imageReady ? api.lectureMediaUrl(current.image_url) : '';
  const counts = presentation?.images;
  const progressText = imageError ? '그림 준비 상태를 확인하지 못했습니다.' : counts?.total
    ? `슬라이드 ${counts.total}장의 설명 그림 · ${counts.ready} / ${counts.total}장 준비 완료`
    : imageBusy ? '강의에 맞는 설명 그림을 구성하고 있습니다.' : '핵심 내용을 슬라이드와 본문으로 살펴보세요.';

  return <div className="presentation-lecture">
    <div className="presentation-status" aria-live="polite">
      <span className={imageBusy && !imageError ? 'is-preparing' : ''}>{progressText}</span>
      {(imageError || presentation?.retryable) && <button type="button" className="text-button" onClick={retryImages}>그림 다시 준비하기</button>}
    </div>
    <div className="lecture-workspace">
      <section className="lecture-stage" aria-label="강의 슬라이드" aria-roledescription="슬라이드 쇼">
        <div className="slide-kicker"><span>개념을 그림으로 이해하기</span><span>{Math.min(slideIndex + 1, slides.length)} / {slides.length}</span></div>
        <div className="slide-content" aria-live="polite">
          <h3>{current?.title}</h3>
          {imageUrl && <button className="slide-image-button" type="button" onClick={() => expandedRef.current?.showModal()} aria-label="설명 그림 크게 보기">
            <img src={imageUrl} alt={current.caption || `${current.title} 설명 그림`} onError={() => setFailedImages(value => ({ ...value, [current.id]: true }))} />
            <span>크게 보기</span>
          </button>}
          {current?.caption && imageUrl && <p className="slide-caption">{current.caption}</p>}
          <ul className={`slide-points${imageUrl ? '' : ' without-image'}`}>{current?.bullets.slice(0, imageUrl ? 2 : 3).map((line, i) => <li key={i}>{line}</li>)}</ul>
          {!imageUrl && ['pending', 'planning', 'generating'].includes(current?.image_status) && <p className="slide-image-note"><span className="presentation-spinner" /> {current?.image_status === 'pending' ? '이 슬라이드의 그림을 순서대로 준비합니다.' : '이 설명에 맞는 그림을 만들고 있습니다.'}</p>}
          {!imageUrl && (current?.image_status === 'failed' || failedImages[current?.id]) && <p className="slide-image-note">그림을 불러오지 못했습니다. 강의 본문으로 계속 공부할 수 있습니다.</p>}
        </div>
        <div className="slide-navigation">
          <button className="btn btn-ghost" type="button" disabled={slideIndex === 0} onClick={() => changeSlide(slideIndex - 1)}>← 이전</button>
          <div className="slide-dots" aria-label="슬라이드 선택">{slides.map((slide, index) => <button key={slide.id} type="button" className={index === slideIndex ? 'is-current' : ''} aria-label={`${index + 1}번 슬라이드 ${slide.title}`} aria-current={index === slideIndex ? 'step' : undefined} onClick={() => changeSlide(index)} />)}</div>
          <button className="btn btn-ghost" type="button" disabled={slideIndex >= slides.length - 1} onClick={() => changeSlide(slideIndex + 1)}>다음 →</button>
        </div>
      </section>
      <section className="lecture-reference" aria-label="전체 강의 본문">
        <div className="reference-heading"><div><span className="reference-kicker">언제든 읽어볼 수 있는</span><h3>강의 본문</h3></div>
          <button className="text-button" type="button" disabled={activeSegment < 0} onClick={() => { setFollow(true); goToParagraph(activeSegment); }}>현재 강의 위치로</button></div>
        <div className="reference-scroll" ref={transcriptRef} onWheel={() => setFollow(false)} onTouchStart={() => setFollow(false)} onKeyDown={event => { if (['PageUp', 'PageDown', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) setFollow(false); }} tabIndex={0}>
          {segments.map((segment, index) => <section key={segment.index} ref={element => { paragraphRefs.current[index] = element; }} className={`reference-paragraph${index === activeSegment ? ' is-speaking' : ''}`} aria-current={index === activeSegment ? 'true' : undefined}>
            <div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{segment.markdown}</ReactMarkdown></div>
            {audio && <button className="paragraph-play text-button" type="button" onClick={() => seekParagraph(index)} aria-label={`${index + 1}번 문단부터 듣기`}>여기부터 듣기</button>}
          </section>)}
          {message.visual && <div className="reference-appendix"><Visual visual={message.visual} /></div>}
          {message.research?.status === 'supplemented' && <Research research={message.research} />}
        </div>
      </section>
    </div>
    <div className="lecture-controls">
      <audio ref={audioRef} src={audio ? api.lectureMediaUrl(audio.audio_url) : undefined} preload="metadata" onTimeUpdate={syncAudio} onSeeking={syncAudio}
        onLoadedMetadata={() => { setDuration(audioRef.current.duration || 0); syncAudio(); }} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)} onError={() => { if (audio) setAudioError('음성 파일을 불러오지 못했습니다. 강의를 다시 열어 주세요.'); }} />
      <button className="btn btn-primary lecture-play" type="button" onClick={startAudio} disabled={audioLoading}>
        {audioLoading ? '음성 준비 중…' : playing ? 'Ⅱ 일시정지' : audio ? '▶ 재생' : '▶ 음성으로 듣기'}</button>
      <div className="lecture-seek"><input type="range" aria-label="강의 재생 위치" min="0" max={duration || 1} step="0.1" value={Math.min(seconds, duration || 1)} disabled={!audio || !duration}
        onChange={event => { audioRef.current.currentTime = Number(event.target.value); syncAudio(); }} /><span>{timeLabel(seconds)} / {timeLabel(duration)}</span></div>
      <label className="lecture-speed"><span>배속</span><select aria-label="강의 재생 속도" value={speed} onChange={event => setSpeed(Number(event.target.value))}>{[0.75, 1, 1.25, 1.5, 2].map(value => <option key={value} value={value}>{value}x</option>)}</select></label>
      <label className="lecture-follow"><input type="checkbox" checked={follow} onChange={event => setFollow(event.target.checked)} />본문 따라가기</label>
      <button className="btn btn-ghost" type="button" onClick={onOpenTutor}>튜터에게 질문</button>
    </div>
    {audioLoading && <p className="lecture-feedback" role="status">음성을 준비하고 있습니다. 슬라이드와 본문은 먼저 읽을 수 있습니다.</p>}
    {audioError && <p className="lecture-feedback is-error" role="alert">{audioError}</p>}
    {(imageError || presentation?.message) && <p className="lecture-feedback" role="status">{imageError || presentation.message}</p>}
    <dialog ref={expandedRef} className="lecture-image-dialog"><button className="btn btn-ghost" type="button" onClick={() => expandedRef.current.close()}>닫기</button>
      {imageUrl && <img src={imageUrl} alt={current.caption || current.title} />}</dialog>
  </div>;
}
