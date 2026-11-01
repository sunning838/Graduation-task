// Immediate readable slides while the image package is being prepared.
export const plainText = text => text.replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1').replace(/[*`>#_~]/g, '').trim();
// Same limit as backend.lecture_presentation.MAX_SLIDES, including provisional slides.
export const MAX_SLIDES = 16;

export function localLecture(text, title = '개념 강의') {
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  const blocks = [];
  let current = [], fence = false;
  const flush = () => { if (current.length) blocks.push(current.join('\n')); current = []; };
  for (const line of lines) {
    if (line.trim().startsWith('```')) fence = !fence;
    if (!fence && !line.trim()) flush(); else current.push(line);
  }
  flush();
  const segments = blocks.map((markdown, index) => ({ markdown, index }));
  return lectureFromSegments(segments, title);
}

export function lectureFromSegments(segments, title = '개념 강의') {
  const slides = [];
  for (const segment of segments) {
    const heading = segment.markdown.match(/^#{1,4}\s+([^\n]+)/);
    if (!slides.length || (heading && slides.length < MAX_SLIDES)) slides.push({ id: `slide-${slides.length}`, title: heading ? plainText(heading[1]) : title,
      bullets: [], segment_indices: [], image_status: 'pending' });
    const slide = slides.at(-1);
    slide.segment_indices.push(segment.index);
    const bullets = segment.markdown.split('\n').filter(line => line.trim() && !/^\s*[#|]|^\s*```/.test(line))
      .map(line => plainText(line.replace(/^\s*(?:[-+*]|\d+[.)])\s+/, ''))).filter(Boolean);
    slide.bullets = [...slide.bullets, ...bullets].slice(0, 3).map(line => line.length > 130 ? `${line.slice(0, 130)}…` : line);
  }
  return { slides, segments };
}

export function slideForSegment(slides, index) {
  return slides.findIndex(slide => slide.segment_indices.includes(index));
}

export function timelineSegment(timeline, seconds) {
  for (let i = timeline.length - 1; i >= 0; i -= 1) {
    if (seconds >= timeline[i].start) return i;
  }
  return 0;
}

export function timeLabel(seconds) {
  const value = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, '0')}`;
}
