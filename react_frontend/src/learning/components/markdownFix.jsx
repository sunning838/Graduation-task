
const BOLD = /\*\*([^*\n[\]]+?)\*\*/g;
const STRONG_HREF = '#ls-strong';

export const fixKoreanBold = text => (text || '').replace(BOLD, (_, inner) => `[${inner}](${STRONG_HREF})`);

export const markdownComponents = {
  a: ({ href, children }) => href === STRONG_HREF
    ? <strong>{children}</strong>
    : <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>,
};