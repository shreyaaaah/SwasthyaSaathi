import React, { useCallback, useEffect, useRef, useState } from 'react';
import { User, Activity, AlertTriangle, PhoneCall } from 'lucide-react';
import SourceCitations from './SourceCitations';

function TriageBanner({ tag, emergencyContact }) {
  if (tag === 'EMERGENCY') return (
    <div role="alert" className="mb-3 p-3 rounded-xl bg-red-950/80 border-2 border-red-500 text-red-100">
      <div className="flex items-start gap-2.5">
        <AlertTriangle aria-hidden="true" className="w-5 h-5 text-red-300 flex-shrink-0" />
        <div className="flex-1">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h4 className="text-sm font-bold">Possible emergency</h4>
            <a href={`tel:${emergencyContact || '112'}`} aria-label="Call emergency services" className="px-3 py-2 rounded-md bg-red-600 hover:bg-red-500 text-white font-bold text-xs flex items-center gap-1">
              <PhoneCall aria-hidden="true" className="w-4 h-4" /> Call {emergencyContact || 'emergency services'}
            </a>
          </div>
          <p className="text-xs mt-1">If you may be in immediate danger, call India’s emergency number or go to the nearest emergency department now. This assistant cannot assess emergencies.</p>
        </div>
      </div>
    </div>
  );
  if (tag === 'URGENT' || tag === 'ROUTINE') return <div role="status" className="mb-3 p-2.5 rounded-xl bg-amber-950/50 border border-amber-500/40 text-amber-100 text-xs">{tag === 'URGENT' ? 'Please seek medical care promptly.' : 'Consider arranging routine medical advice if symptoms continue.'}</div>;
  return null;
}

function renderInlineMarkdown(text, keyPrefix) {
  const pattern = /(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`\n]+`)/g;
  return String(text).split(pattern).map((part, index) => {
    const key = `${keyPrefix}-inline-${index}`;
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={key} className="font-semibold text-slate-50">{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith('*') && part.endsWith('*')) return <em key={key}>{part.slice(1, -1)}</em>;
    if (part.startsWith('`') && part.endsWith('`')) return <code key={key} className="rounded bg-slate-800 px-1 py-0.5">{part.slice(1, -1)}</code>;
    return <React.Fragment key={key}>{part}</React.Fragment>;
  });
}

function renderAssistantMarkdown(text, keyPrefix) {
  const lines = String(text || '').split(/\r?\n/);
  const blocks = [];
  let lineIndex = 0;

  while (lineIndex < lines.length) {
    const line = lines[lineIndex].trim();
    if (!line) {
      lineIndex += 1;
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+?)\s*#*$/);
    if (heading) {
      const Heading = heading[1].length === 1 ? 'h2' : 'h3';
      blocks.push(<Heading key={`${keyPrefix}-heading-${lineIndex}`} className="text-sm font-semibold text-slate-50">{renderInlineMarkdown(heading[2], `${keyPrefix}-${lineIndex}`)}</Heading>);
      lineIndex += 1;
      continue;
    }

    const listItem = line.match(/^(?:([-+*•])\s+|(\d+)[.)]\s+)(.+)$/);
    if (listItem) {
      const ordered = Boolean(listItem[2]);
      const items = [];
      const firstNumber = ordered ? Number(listItem[2]) : undefined;
      while (lineIndex < lines.length) {
        const item = lines[lineIndex].trim().match(/^(?:([-+*•])\s+|(\d+)[.)]\s+)(.+)$/);
        if (!item || Boolean(item[2]) !== ordered) break;
        items.push(item[3]);
        lineIndex += 1;
      }
      const List = ordered ? 'ol' : 'ul';
      blocks.push(
        <List key={`${keyPrefix}-list-${lineIndex}`} start={firstNumber} className={`${ordered ? 'list-decimal' : 'list-disc'} space-y-1 pl-5 marker:text-teal-300`}>
          {items.map((item, itemIndex) => <li key={`${keyPrefix}-item-${itemIndex}`} className="pl-1">{renderInlineMarkdown(item, `${keyPrefix}-item-${lineIndex}-${itemIndex}`)}</li>)}
        </List>,
      );
      continue;
    }

    const paragraph = [line];
    lineIndex += 1;
    while (lineIndex < lines.length) {
      const next = lines[lineIndex].trim();
      if (!next || /^(?:#{1,3}\s+|(?:[-+*•])\s+|\d+[.)]\s+)/.test(next)) break;
      paragraph.push(next);
      lineIndex += 1;
    }
    blocks.push(<p key={`${keyPrefix}-paragraph-${lineIndex}`}>{renderInlineMarkdown(paragraph.join(' '), `${keyPrefix}-${lineIndex}`)}</p>);
  }

  return <div className="space-y-2.5">{blocks}</div>;
}

function messageKey(message, index) {
  return message.id || `${message.sender}-${index}`;
}

const MessageList = ({ messages, isLoading, loadingMessage }) => {
  const containerRef = useRef(null);
  const previousMessageCount = useRef(messages.length);
  const pendingAnimationFrame = useRef(null);
  const [handledAnimationIds, setHandledAnimationIds] = useState(() => new Set());
  const [animation, setAnimation] = useState(null);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(() => (
    typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  ));

  const latestMessage = messages[messages.length - 1];
  const latestMessageId = latestMessage ? messageKey(latestMessage, messages.length - 1) : null;
  const latestWords = latestMessage?.sender === 'bot' && latestMessage.animate
    ? String(latestMessage.text || '').match(/\S+\s*/g) || []
    : [];
  const shouldAnimateLatest = Boolean(
    latestMessageId
    && latestWords.length
    && !prefersReducedMotion
    && latestMessage?.sender === 'bot'
    && latestMessage.animate
    && latestMessage.triage_tag !== 'EMERGENCY'
    && !handledAnimationIds.has(latestMessageId)
    && animation?.messageId !== latestMessageId
  );

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return undefined;
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const updatePreference = () => setPrefersReducedMotion(media.matches);
    media.addEventListener?.('change', updatePreference);
    return () => media.removeEventListener?.('change', updatePreference);
  }, []);

  useEffect(() => {
    if (!latestMessage || latestMessage.sender !== 'bot' || !latestMessage.animate || !latestMessageId) return;
    if (prefersReducedMotion || latestMessage.triage_tag === 'EMERGENCY') return;
    if (handledAnimationIds.has(latestMessageId)) return;

    const words = String(latestMessage.text || '').match(/\S+\s*/g) || [];
    if (!words.length) return;

    const frame = window.requestAnimationFrame(() => {
      setHandledAnimationIds((current) => new Set(current).add(latestMessageId));
      setAnimation({ messageId: latestMessageId, words, visibleWords: 0 });
      if (pendingAnimationFrame.current === frame) pendingAnimationFrame.current = null;
    });
    pendingAnimationFrame.current = frame;
    return () => {
      window.cancelAnimationFrame(frame);
      if (pendingAnimationFrame.current === frame) pendingAnimationFrame.current = null;
    };
  }, [handledAnimationIds, latestMessage, latestMessageId, prefersReducedMotion]);

  const finishTyping = useCallback((messageId, text) => {
    if (pendingAnimationFrame.current !== null) {
      window.cancelAnimationFrame(pendingAnimationFrame.current);
      pendingAnimationFrame.current = null;
    }
    setHandledAnimationIds((current) => new Set(current).add(messageId));
    const words = String(text || '').match(/\S+\s*/g) || [];
    setAnimation({ messageId, words, visibleWords: words.length });
  }, []);

  useEffect(() => {
    if (!animation) return undefined;
    if (animation.visibleWords >= animation.words.length) {
      const finishTimer = window.setTimeout(() => {
        setAnimation((current) => current?.messageId === animation.messageId ? null : current);
      }, 150);
      return () => window.clearTimeout(finishTimer);
    }

    const delay = Math.max(8, Math.min(28, 2800 / Math.max(1, animation.words.length)));
    const timer = window.setTimeout(() => {
      setAnimation((current) => current?.messageId === animation.messageId
        ? { ...current, visibleWords: current.visibleWords + 1 }
        : current);
    }, delay);
    return () => window.clearTimeout(timer);
  }, [animation]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const hasNewMessage = previousMessageCount.current !== messages.length;
    const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 140;
    if (hasNewMessage || isLoading || nearBottom) container.scrollTop = container.scrollHeight;
    previousMessageCount.current = messages.length;
  }, [messages.length, isLoading, animation?.visibleWords]);

  useEffect(() => {
    if (!animation && !shouldAnimateLatest) return undefined;
    const finishOnEscape = (event) => {
      if (event.key !== 'Escape') return;
      const messageId = animation?.messageId || latestMessageId;
      const message = animation
        ? messages.find((item, index) => messageKey(item, index) === messageId)
        : latestMessage;
      if (messageId && message) finishTyping(messageId, message.text);
    };
    window.addEventListener('keydown', finishOnEscape);
    return () => window.removeEventListener('keydown', finishOnEscape);
  }, [animation, finishTyping, latestMessage, latestMessageId, messages, shouldAnimateLatest]);

  return (
    <div ref={containerRef} className="flex-1 overflow-y-auto p-4 space-y-4" aria-live={animation ? 'off' : 'polite'} aria-busy={Boolean(isLoading || animation)}>
      {messages.map((message, index) => {
        const isUser = message.sender === 'user';
        const id = messageKey(message, index);
        const isTyping = id === latestMessageId
          && !prefersReducedMotion
          && (animation?.messageId === id || shouldAnimateLatest);
        const words = animation?.messageId === id ? animation.words : (isTyping ? latestWords : []);
        const visibleWords = animation?.messageId === id ? animation.visibleWords : 0;
        const visibleText = isTyping ? words.slice(0, visibleWords).join('') : message.text;

        return (
          <div key={id} className={`flex items-start gap-3 animate-fade-in ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
            <div aria-hidden="true" className={`w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 shadow-md ${isUser ? 'bg-gradient-to-tr from-sky-500 to-indigo-500 text-white' : 'bg-gradient-to-tr from-teal-500 to-emerald-500 text-slate-950 font-bold'}`}>
              {isUser ? <User className="w-5 h-5" /> : <Activity className="w-5 h-5" />}
            </div>
            <div className={`flex max-w-[88%] md:max-w-[80%] flex-col gap-2 ${isUser ? 'items-end' : 'items-start'}`}>
              <div className={`w-fit max-w-full rounded-2xl p-4 shadow-lg leading-relaxed text-sm ${isUser ? 'bg-gradient-to-r from-teal-600 to-cyan-700 text-white rounded-tr-none' : 'bg-slate-900/90 border border-slate-700/80 text-slate-100 rounded-tl-none'}`}>
                <div className="flex items-center justify-between gap-3 mb-2 text-xs opacity-80">
                  <span className="font-semibold tracking-wide">{isUser ? 'You' : 'SwasthyaSaathi'}</span>
                  {!isUser && <div className="flex items-center gap-2">
                    {isTyping && <button type="button" onClick={() => finishTyping(id, message.text)} className="text-[10px] text-slate-300 hover:text-white underline underline-offset-2">Show full reply</button>}
                    {message.answer_mode && message.answer_mode !== 'SOCIAL' && <span className="text-[10px] text-teal-300">{{ GROUNDED: 'From MoHFW/ICMR/WHO guidelines', WEB_TRUSTED: 'From trusted health websites', MIXED: 'Guidelines + general information', GENERAL: 'General health information' }[message.answer_mode] || ''}</span>}
                  </div>}
                </div>
                {!isUser && ['EMERGENCY', 'URGENT', 'ROUTINE'].includes(message.triage_tag) && <TriageBanner tag={message.triage_tag} emergencyContact={message.emergency_contact} />}
                <div className="text-sm leading-relaxed">
                  {isUser
                    ? <div className="whitespace-pre-wrap">{message.text}</div>
                    : <div className="relative">
                      {renderAssistantMarkdown(visibleText, id)}
                      {isTyping && <span aria-hidden="true" className="inline-block h-4 ml-0.5 align-middle border-r-2 border-teal-300 animate-pulse" />}
                    </div>}
                </div>
              </div>
              {!isUser && !isTyping && message.sources?.length > 0 && <SourceCitations sources={message.sources} isGrounded={message.is_grounded} />}
            </div>
          </div>
        );
      })}
      {isLoading && <div role="status" className="flex items-center gap-3 p-4 text-slate-300 text-sm"><Activity aria-hidden="true" className="w-5 h-5 text-teal-400 animate-spin" /><span>{loadingMessage || 'Looking for relevant health guidance…'}</span></div>}
    </div>
  );
};

export default MessageList;
