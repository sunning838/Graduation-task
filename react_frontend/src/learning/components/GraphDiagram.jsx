import { useId } from 'react';

// Render validated graph structure as SVG text/shapes, never raw model HTML/DOT.
export default function GraphDiagram({ visual }) {
  const marker = useId().replace(/:/g, '');
  if (!Array.isArray(visual.nodes) || !Array.isArray(visual.edges) || !visual.nodes.length) {
    return <p className="muted">도식을 표시하지 못했습니다. 본문 설명을 참고해 주세요.</p>;
  }
  const nodes = visual.nodes;
  const edges = visual.edges.filter(e => Number.isInteger(e.from) && Number.isInteger(e.to) && e.from >= 0 && e.to >= 0 && e.from < nodes.length && e.to < nodes.length);
  const hierarchy = visual.layout === 'hierarchy';
  const levels = nodes.map(() => 0);
  if (hierarchy) for (let pass = 0; pass < nodes.length; pass += 1) {
    edges.forEach(e => { levels[e.to] = Math.min(nodes.length - 1, Math.max(levels[e.to], levels[e.from] + 1)); });
  }
  const widestLevel = Math.max(...levels.map(level => levels.filter(v => v === level).length));
  const width = Math.max(320, (hierarchy ? widestLevel : nodes.length) * 240);
  const top = 75 + edges.length * 24;
  const boxes = nodes.map((node, i) => {
    const lines = String(node).match(/.{1,14}/gu) || [''];
    const peers = levels.map((level, j) => level === levels[i] ? j : -1).filter(j => j >= 0);
    return { lines, height: Math.max(70, lines.length * 20 + 20),
      x: hierarchy ? (width / peers.length) * (peers.indexOf(i) + .5) : i * 240 + 120,
      y: hierarchy ? 20 + levels[i] * 190 : top };
  });
  const height = Math.max(...boxes.map(b => b.y + b.height)) + 40;
  return <div style={{ overflowX: 'auto' }}>
    <svg role="img" aria-label={visual.title} width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <title>{visual.title}</title>
      <defs><marker id={marker} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#5978ae" /></marker></defs>
      {edges.map((edge, i) => {
        const fromBox = boxes[edge.from], toBox = boxes[edge.to];
        const from = fromBox.x, to = toBox.x;
        const y = 30 + i * 24;
        const loop = edge.from === edge.to;
        const label = `${edge.relation || ''}${edge.label ? ': ' + edge.label : ''}`;
        return <g key={i}>
          <path d={hierarchy ? `M${from},${fromBox.y + fromBox.height} L${to},${toBox.y}` : loop ? `M${from - 40},${top} C${from - 120},${y} ${from + 120},${y} ${from + 40},${top}` : `M${from},${top} C${from},${y} ${to},${y} ${to},${top}`}
            fill="none" stroke="#5978ae" strokeWidth="1.6" markerEnd={`url(#${marker})`} />
          <text x={(from + to) / 2} y={hierarchy ? (fromBox.y + fromBox.height + toBox.y) / 2 : y + 12} textAnchor="middle" fontSize="12" fill="currentColor">{label}<title>{label}</title></text>
        </g>;
      })}
      {boxes.map((box, i) => {
        return <g key={i}><rect x={box.x - 100} y={box.y} width="200" height={box.height} rx="12" fill="#eaf3ff" stroke="#5978ae" />
          <text x={box.x} y={box.y + 27} textAnchor="middle" fill="#153352" fontSize="14">{box.lines.map((line, j) => <tspan key={j} x={box.x} dy={j ? 20 : 0}>{line}</tspan>)}</text></g>;
      })}
    </svg>
  </div>;
}
