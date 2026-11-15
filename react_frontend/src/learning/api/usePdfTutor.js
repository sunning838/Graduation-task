import { useEffect, useRef, useState } from 'react';
import { askPdf } from './api.js';

export default function usePdfTutor(record, answerLength) {
  const [chats, setChats] = useState({});
  const requests = useRef(new Map());
  useEffect(() => {
    const active = requests.current;
    return () => {
      for (const controller of active.values()) controller.abort();
      active.clear();
    };
  }, []);

  const chat = record ? chats[record.id] : null;
  const messages = chat?.messages ?? [];
  const ready = Boolean(record?.sourcePages?.length);
  const send = async question => {
    if (!ready || requests.current.has(record.id)) return false;
    const { id } = record;
    const controller = new AbortController();
    requests.current.set(id, controller);
    const update = state => setChats(current => ({ ...current, [id]: state }));
    const userMessage = { id: crypto.randomUUID(), role: 'user', content: question };
    update({ messages, pending: userMessage });
    try {
      const history = messages.slice(-6).map(message => ({ role: message.role, text: message.content.slice(0, 2000) }));
      const data = await askPdf(record, question, history, answerLength, controller.signal);
      if (controller.signal.aborted) return false;
      update({ messages: [...messages, userMessage, {
        id: crypto.randomUUID(), role: 'assistant', content: data.answer, research: data.research,
      }] });
      return true;
    } catch (error) {
      if (!controller.signal.aborted) update({ messages, error: error.message });
      return false;
    } finally {
      if (requests.current.get(id) === controller) requests.current.delete(id);
    }
  };

  return {
    messages: chat?.pending ? [...messages, chat.pending] : messages,
    loading: Boolean(chat?.pending),
    error: chat?.error,
    ready,
    send,
  };
}
