import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import * as api from '../api/lessonClient.js';
import { Visual, Research } from './ChatMessage.jsx';
import { localLecture, lectureFromSegments, slideForSegment, timelineSegment, timeLabel } from './lectureSlides.js';
import { fixKoreanBold, markdownComponents } from './markdownFix.jsx';
import { ChatIcon, ChevronLeft, ChevronRight, ExpandIcon, ImageIcon, PauseIcon, PlayIcon } from './icons.jsx';

const SETTLED = new Set(['idle', 'ready', 'partial', 'failed']);
const SPEEDS = [0.75, 1, 1.25, 1.5, 2];

export default function PresentationLecture({ lessonId, variant, message, onOpenTutor }) {
  const local = useMemo(() => localLecture(message.text), [message.text]);
  const [presentation, setPresentation] = useState(null);
  const [imageError, setImageError] = useState('');
  const [imageReload, setImageReload] = useState(0);
  const [imageSubmitting, setImageSubmitting] = useState(false);
  const [audio, setAudio] = useState(null);
  const [audioLoading, setAudioLoading] = useState(false);
  const [audioError, setAudioError] = useState('');
  const [playing, setPlaying] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [duration, setDuration] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [slideIndex, setSlideIndex] = useState(0);
  const [activeSegment, setActiveSegment] = useState(-1);
  // 본문 따라가기: 기본으로 켜 두고, 사용자가 본문을 직접 스크롤하면 잠시 끕니다.
  // 재생·슬라이드 이동·문단 듣기를 누르면 다시 켜집니다.
  const [follow, setFollow] = useState(true);
  const [failedImages, setFailedImages] = useState({});
  const audioRef = useRef(null);
  const transcriptRef = useRef(null);
  const paragraphRefs = useRef([]);
  const playRequested = useRef(false);
  const requestedSegment = useRef(0);
  const audioController = useRef(null);
  const imageRequest = useRef(null);
  const expandedRef = useRef(null);
  const fallback = useMemo(() => audio?.timeline ? lectureFromSegments(audio.timeline) : local, [audio, local]);
  const slides = presentation?.slides?.length ? presentation.slides : fallback.slides;
  const segments = audio?.timeline || presentation?.segments || local.segments;
  const current = slides[Math.min(slideIndex, slides.length - 1)];
  const packageId = presentation?.package_id;
  const activePackageId = presentation && !SETTLED.has(presentation.status) ? packageId : null;

  useEffect(() => {
    if (packageId && !activePackageId) return;
    const controller = new AbortController();
    let timer;
    let failures = 0;
    const checkImages = async () => {
      try {
        const result = activePackageId
          ? await api.getPresentation(activePackageId, controller.signal)
          : await api.lookupPresentation(lessonId, variant, message.text, controller.signal);
        if (controller.signal.aborted) return;
        setPresentation(result); setImageError(''); failures = 0;
        if (activePackageId && !SETTLED.has(result.status)) timer = setTimeout(checkImages, 1800);
      } catch (error) {
        if (controller.signal.aborted) return;
        if (activePackageId && ++failures < 3) timer = setTimeout(checkImages, 2500);
        else setImageError(error.message);
      }
    };
    if (activePackageId) timer = setTimeout(checkImages, 1500);
    else checkImages();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [lessonId, variant, message.text, imageReload, packageId, activePackageId]);

  useEffect(() => () => {
    audioController.current?.abort();
    imageRequest.current?.abort();
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
    box.scrollTo({ top: Math.max(0, paragraph.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop - 24), behavior: 'smooth' });
  };

  useEffect(() => {
    if (!follow || activeSegment < 0) return;
    const box = transcriptRef.current, paragraph = paragraphRefs.current[activeSegment];
    if (box && paragraph) box.scrollTo({ top: Math.max(0, paragraph.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop - 24), behavior: 'smooth' });
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
    setFollow(true);
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
    setFollow(true);
    const target = slides[index].segment_indices[0];
    if (audio && audioRef.current) {
      const item = audio.timeline.find(seg => seg.index === target);
      if (item) { audioRef.current.currentTime = item.start; setSeconds(item.start); setActiveSegment(target); }
    }
    goToParagraph(target);
  };

  const seekParagraph = index => {
    if (!audioRef.current || !audio) return;
    setFollow(true);
    audioRef.current.currentTime = audio.timeline[index].start;
    syncAudio();
    audioRef.current.play().catch(() => setAudioError('재생 버튼을 눌러 주세요.'));
  };

  const requestImages = async () => {
    if (imageRequest.current || !presentation || (presentation.status !== 'idle' && !presentation.retryable)) return;
    const controller = new AbortController();
    imageRequest.current = controller;
    setImageSubmitting(true);
    setImageError('');
    setFailedImages({});
    try {
      const result = presentation.retryable
        ? await api.retryPresentation(packageId, controller.signal)
        : await api.startPresentation(lessonId, variant, message.text, controller.signal);
      if (!controller.signal.aborted) setPresentation(result);
    } catch (error) {
      if (!controller.signal.aborted) setImageError(error.message);
    } finally {
      imageRequest.current = null;
      if (!controller.signal.aborted) setImageSubmitting(false);
    }
  };

  const imageReady = current?.image_status === 'ready' && !failedImages[current.id];
  const imageUrl = imageReady ? api.lectureMediaUrl(current.image_url) : '';
  const canCreateImages = presentation?.status === 'idle' || presentation?.retryable;
  const imagePreparing = !imageUrl && activePackageId && !imageError && ['pending', 'planning', 'generating'].includes(current?.image_status);
  const imageFailed = !imageUrl && (current?.image_status === 'failed' || failedImages[current?.id]);
  const slideNumber = Math.min(slideIndex + 1, slides.length);

  // 그림 자리에 무엇을 보여줄지 한 곳에서 결정합니다.
  let figure = null;
  if (imageUrl) {
    figure = <button className="ls-figure-button" type="button" onClick={() => expandedRef.current?.showModal()} aria-label="설명 그림 크게 보기">
      <img src={imageUrl} alt={current.caption || `${current.title} 설명 그림`} onError={() => setFailedImages(value => ({ ...value, [current.id]: true }))} />
      <span className="ls-figure-zoom"><ExpandIcon size={13} /></span>
    </button>;
  } else if (canCreateImages) {
    figure = <button className="ls-pill ls-pill-strong" type="button" onClick={requestImages} disabled={imageSubmitting}>
      <ImageIcon />{imageSubmitting ? '그림 준비 중…' : presentation?.retryable ? '그림 다시 만들기' : '설명 그림 만들기'}
    </button>;
  } else if (imagePreparing) {
    figure = <p className="ls-figure-note"><span className="presentation-spinner" />{current?.image_status === 'pending' ? '순서대로 그림을 준비하고 있어요' : '그림을 만들고 있어요'}</p>;
  } else if (imageFailed) {
    figure = <p className="ls-figure-note">그림을 불러오지 못했어요</p>;
  } else if (imageError) {
    figure = <button className="ls-pill" type="button" onClick={() => { setImageError(''); setImageReload(value => value + 1); }}>그림 상태 다시 확인</button>;
  }

  return <div className="ls-lecture">
    <div className="ls-grid">
      <section className="ls-card ls-slide" aria-label="강의 슬라이드" aria-roledescription="슬라이드 쇼">
        <div className="ls-slide-head">
          <h2>{current?.title}</h2>
          <span className="ls-chip">{slideNumber} / {slides.length}</span>
        </div>
        <div className={`ls-figure${imageUrl ? ' has-image' : ''}`}>{figure}</div>
        {current?.caption && imageUrl && <p className="ls-caption">{current.caption}</p>}
        <ul className="ls-points">{current?.bullets.slice(0, imageUrl ? 2 : 3).map((line, i) => <li key={i}>{line}</li>)}</ul>
        <div className="ls-slide-nav">
          <button className="ls-round" type="button" aria-label="이전 슬라이드" disabled={slideIndex === 0} onClick={() => changeSlide(slideIndex - 1)}><ChevronLeft /></button>
          <div className="ls-dots" aria-label="슬라이드 선택">{slides.map((slide, index) => <button key={slide.id} type="button" className={index === slideIndex ? 'is-current' : ''} aria-label={`${index + 1}번 슬라이드 ${slide.title}`} aria-current={index === slideIndex ? 'step' : undefined} onClick={() => changeSlide(index)} />)}</div>
          <button className="ls-round" type="button" aria-label="다음 슬라이드" disabled={slideIndex >= slides.length - 1} onClick={() => changeSlide(slideIndex + 1)}><ChevronRight /></button>
        </div>
      </section>

      <section className="ls-card ls-reference" aria-label="강의 본문">
        <div className="ls-reference-head">
          <h2>강의 본문</h2>
          <button className="ls-pill" type="button" onClick={onOpenTutor}><ChatIcon />튜터에게 질문</button>
        </div>
        <div className="ls-reference-scroll" ref={transcriptRef} onWheel={() => setFollow(false)} onTouchStart={() => setFollow(false)} onKeyDown={event => { if (['PageUp', 'PageDown', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) setFollow(false); }} tabIndex={0}>
          {segments.map((segment, index) => <section key={segment.index} ref={element => { paragraphRefs.current[index] = element; }} className={`reference-paragraph${index === activeSegment ? ' is-speaking' : ''}`} aria-current={index === activeSegment ? 'true' : undefined}>
            <div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{fixKoreanBold(segment.markdown)}</ReactMarkdown></div>
            {audio && <button className="paragraph-play text-button" type="button" onClick={() => seekParagraph(index)} aria-label={`${index + 1}번 문단부터 듣기`}>여기부터 듣기</button>}
          </section>)}
          {message.visual && <div className="reference-appendix"><Visual visual={message.visual} /></div>}
          {message.research?.status === 'supplemented' && <Research research={message.research} />}
        </div>
        <div className="ls-player">
          <audio ref={audioRef} src={audio ? api.lectureMediaUrl(audio.audio_url) : undefined} preload="metadata" onTimeUpdate={syncAudio} onSeeking={syncAudio}
            onLoadedMetadata={() => { setDuration(audioRef.current.duration || 0); syncAudio(); }} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
            onEnded={() => setPlaying(false)} onError={() => { if (audio) setAudioError('음성 파일을 불러오지 못했습니다. 강의를 다시 열어 주세요.'); }} />
          <button className="ls-play" type="button" onClick={startAudio} disabled={audioLoading}
            aria-label={audioLoading ? '음성 준비 중' : playing ? '일시정지' : '음성으로 듣기'}>
            {audioLoading ? <span className="ls-play-spinner" /> : playing ? <PauseIcon /> : <PlayIcon />}
          </button>
          <div className="ls-seek">
            <input type="range" aria-label="강의 재생 위치" min="0" max={duration || 1} step="0.1" value={Math.min(seconds, duration || 1)} disabled={!audio || !duration}
              onChange={event => { audioRef.current.currentTime = Number(event.target.value); syncAudio(); }} />
            <div className="ls-times"><span>{timeLabel(seconds)}</span><span>{timeLabel(duration)}</span></div>
          </div>
          <select className="ls-speed" aria-label="강의 재생 속도" value={speed} onChange={event => setSpeed(Number(event.target.value))}>
            {SPEEDS.map(value => <option key={value} value={value}>{value}x</option>)}
          </select>
        </div>
      </section>
    </div>

    {audioLoading && <p className="ls-feedback" role="status">음성을 준비하고 있어요. 그동안 슬라이드와 본문을 먼저 읽을 수 있어요.</p>}
    {audioError && <p className="ls-feedback is-error" role="alert">{audioError}</p>}
    {(imageError || presentation?.message) && <p className="ls-feedback" role="status">{imageError || presentation.message}</p>}

    <dialog ref={expandedRef} className="lecture-image-dialog ls-image-dialog">
      <button className="ls-round" type="button" aria-label="닫기" onClick={() => expandedRef.current.close()}>×</button>
      {imageUrl && <img src={imageUrl} alt={current.caption || current.title} />}
    </dialog>
  </div>;
}