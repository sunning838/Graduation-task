import { useEffect, useRef } from 'react';

export default function TutorDrawer({ open, onClose, children, heading = '강의 중 궁금한 점을 질문해 보세요.' }) {
  const dialog = useRef(null);
  useEffect(() => {
    const element = dialog.current;
    if (open && !element.open) element.showModal();
    if (!open && element.open) element.close();
  }, [open]);
  return <dialog ref={dialog} className="lecture-tutor-drawer" onCancel={onClose} onClose={onClose} aria-label="튜터에게 질문">
    <div className="tutor-drawer-heading"><span>{heading}</span><button type="button" className="btn btn-ghost" onClick={onClose}>닫기</button></div>
    {children}
  </dialog>;
}