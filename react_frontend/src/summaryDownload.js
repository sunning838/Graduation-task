export function summaryMarkdown(label, notes) {
  return `# ${label} 취약점 개념 요약\n\n` + notes.map(note =>
    `## ${note.label}\n\n생성 시각: ${note.created_at}\n\n` +
    `생성 당시: ${note.basis.correct}/${note.basis.total}문제 정답 · ${note.basis.accuracy}%\n\n` +
    (note.stale ? '> 현재 기록 또는 자료와 달라 갱신이 필요한 요약입니다.\n\n' : '') +
    note.markdown
  ).join('\n\n---\n\n');
}

export function downloadSummary(label, notes) {
  const url = URL.createObjectURL(new Blob([summaryMarkdown(label, notes)], { type: 'text/markdown;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `${label}_취약점_개념요약.md`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
