
const base = { fill: 'none', stroke: 'currentColor', strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true };

export const ChevronLeft = ({ size = 16 }) => <svg width={size} height={size} viewBox="0 0 24 24" strokeWidth="2.1" {...base}><path d="M15 18l-6-6 6-6" /></svg>;
export const ChevronRight = ({ size = 16 }) => <svg width={size} height={size} viewBox="0 0 24 24" strokeWidth="2.1" {...base}><path d="M9 6l6 6-6 6" /></svg>;
export const PencilIcon = ({ size = 16 }) => <svg width={size} height={size} viewBox="0 0 24 24" strokeWidth="1.9" {...base}><path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z" /></svg>;
export const ListIcon = ({ size = 16 }) => <svg width={size} height={size} viewBox="0 0 24 24" strokeWidth="1.8" {...base}><path d="M8 6h12M8 12h12M8 18h12" /><circle cx="4" cy="6" r="0.7" fill="currentColor" /><circle cx="4" cy="12" r="0.7" fill="currentColor" /><circle cx="4" cy="18" r="0.7" fill="currentColor" /></svg>;
export const ImageIcon = ({ size = 17 }) => <svg width={size} height={size} viewBox="0 0 24 24" strokeWidth="1.8" {...base}><rect x="3" y="4" width="18" height="16" rx="4" /><circle cx="9" cy="10" r="1.5" /><path d="M21 16l-5-5-8 9" /></svg>;
export const ChatIcon = ({ size = 15 }) => <svg width={size} height={size} viewBox="0 0 24 24" strokeWidth="1.9" {...base}><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z" /></svg>;
export const CloseIcon = ({ size = 16 }) => <svg width={size} height={size} viewBox="0 0 24 24" strokeWidth="2" {...base}><path d="M6 6l12 12M18 6L6 18" /></svg>;
export const ExpandIcon = ({ size = 15 }) => <svg width={size} height={size} viewBox="0 0 24 24" strokeWidth="1.9" {...base}><path d="M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7" /></svg>;
export const PlayIcon = ({ size = 16 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5.5v13a1 1 0 0 0 1.5.9l10.4-6.5a1 1 0 0 0 0-1.8L9.5 4.6A1 1 0 0 0 8 5.5z" /></svg>;
export const PauseIcon = ({ size = 16 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="5" width="4" height="14" rx="1.2" /><rect x="14" y="5" width="4" height="14" rx="1.2" /></svg>;