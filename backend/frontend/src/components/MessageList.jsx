import React, { useCallback, useEffect, useState } from 'react';
import { User, Activity, AlertTriangle, PhoneCall } from 'lucide-react';
import SourceCitations from './SourceCitations';

const messageKey = (message, index) => message.id || `${message.sender}-${index}`;

function normalizeAssistantAnswer(text = '') {
  let answer = String(text).replace(/\r\n?/g, '\n').trim();
  answer = answer
    .replace(/(^|\n)[ \t]*\*[ \t]+(?=\*\*|[A-Z0-9])/g, '$1- ')
    .replace(/([.!?)])\*[ \t]+(?=\*\*|[A-Z0-9])/g, '$1\n- ')
    .replace(/[ \t]+\*[ \t]+(?=\*\*|[A-Z0-9])/g, '\n- ');
  if (/:[ \t]*-[ \t]+/.test(answer)) {
    answer = answer
      .replace(/:[ \t]*-[ \t]+/, ':\n- ')
      .replace(/[ \t]+-[ \t]+(?=\*\*|[A-Z0-9])/g, '\n- ');
  }
  return answer.replace(
    /([.!?)])[ \t]+(?=(?:If you notice|If you experience|If symptoms|If the fever|Please seek|Seek medical|Get medical|Contact a health|A health professional)\b)/g,
    '$1\n\n',
  );
}

function splitIntoBubbles(text) {
  const paragraphs = normalizeAssistantAnswer(text).split(/\n\s*\n/).map((part) => part.trim()).filter(Boolean);
  if (paragraphs.length <= 3) return paragraphs;
  return [paragraphs[0], paragraphs[1], paragraphs.slice(2).join('\n\n')];
}

function renderInlineMarkdown(text, keyPrefix) {
  const parts = String(text).split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g);
  return parts.map((part, index) => {
    const key = `${keyPrefix}-inline-${index}`;
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={key} className="font-semibold text-slate-50">{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith('*') && part.endsWith('*')) {
      return <em key={key}>{part.slice(1, -1)}</em>;
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={key} className="rounded bg-slate-800 px-1 py-0.5">{part.slice(1, -1)}</code>;
    }
    return <React.Fragment key={key}>{part.replace(/^#{1,6}\s*/, '').replace(/\*\*?|`/g, '')}</React.Fragment>;
  });
}

function renderAssistantMarkdown(text, keyPrefix) {
  const lines = String(text).split('\n');
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
      blocks.push(<Heading key={`${keyPrefix}-block-${lineIndex}`} className="text-sm font-semibold text-slate-50">{renderInlineMarkdown(heading[2], `${keyPrefix}-${lineIndex}`)}</Heading>);
      lineIndex += 1;
      continue;
    }

    const firstListItem = line.match(/^(?:([-+*])\s+|(\d+)[.)]\s+)(.+)$/);
    if (firstListItem) {
      const ordered = Boolean(firstListItem[2]);
      const items = [];
      while (lineIndex < lines.length) {
        const item = lines[lineIndex].trim().match(/^(?:([-+*])\s+|(\d+)[.)]\s+)(.+)$/);
        if (!item || Boolean(item[2]) !== ordered) break;
        items.push(item[3]);
        lineIndex += 1;
      }
      const List = ordered ? 'ol' : 'ul';
      blocks.push(
        <List key={`${keyPrefix}-list-${lineIndex}`} className={`${ordered ? 'list-decimal' : 'list-disc'} space-y-1 pl-5 marker:text-teal-300`}>
          {items.map((item, itemIndex) => <li key={`${keyPrefix}-item-${itemIndex}`} className="whitespace-pre-wrap">{renderInlineMarkdown(item, `${keyPrefix}-item-${lineIndex}-${itemIndex}`)}</li>)}
        </List>,
      );
      continue;
    }

    const paragraph = [line];
    lineIndex += 1;
    while (lineIndex < lines.length) {
      const next = lines[lineIndex].trim();
      if (!next || /^(?:#{1,3}\s+|(?:[-+*])\s+|\d+[.)]\s+)/.test(next)) break;
      paragraph.push(next);
      lineIndex += 1;
    }
    blocks.push(<p key={`${keyPrefix}-paragraph-${lineIndex}`} className="whitespace-pre-wrap">{renderInlineMarkdown(paragraph.join(' '), `${keyPrefix}-${lineIndex}`)}</p>);
  }

  return <div className="space-y-2.5">{blocks}</div>;
}

function TypingDots({ label = 'SwasthyaSaathi is typing' }) {
  return (
    <span className="typing-dots" role="status" aria-label={label}>
      <span /><span /><span />
    </span>
  );
}

function TriageBanner({ tag, emergencyContact }) {
  if (tag === 'EMERGENCY') return (
    <div role="alert" className="mb-3 p-3 rounded-xl bg-red-950/80 border-2 border-red-500 text-red-100">
      <div className="flex items-start gap-2.5">
        <AlertTriangle aria-hidden="true" className="w-5 h-5 text-red-300 flex-shrink-0" />
        <div className="flex-1">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h4 className="text-sm font-bold">Possible emergency</h4>
            <a href={`tel:${emergencyContact || '112'}`} aria-label="Call emergency services" className="px-3 py-2 rounded-md bg-red-600 hover:bg-red-500 text-white font-bold text-xs flex items-center gap-1">
              <PhoneCall aria-hidden="true" className="w-4 h-4" /> Call {emergencyContact || '112'}
            </a>
          </div>
          <p className="text-xs mt-1">If you may be in immediate danger, call India’s emergency number or go to the nearest emergency department now. This assistant cannot assess emergencies.</p>
        </div>
      </div>
    </div>
  );
  if (['URGENT', 'CONSULT_SOON'].includes(tag)) return (
    <div role="status" className="mb-3 p-2.5 rounded-xl bg-amber-950/50 border border-amber-500/40 text-amber-100 text-xs">
      Please seek medical care promptly.
    </div>
  );
  if (tag === 'ROUTINE') return (
    <div role="status" className="mb-3 p-2.5 rounded-xl bg-amber-950/50 border border-amber-500/40 text-amber-100 text-xs">
      Consider arranging routine medical advice if symptoms continue.
    </div>
  );
  return null;
}

function getWords(text) {
  return text.match(/\S+\s*/g) || [];
}

const MessageList = ({ messages, isLoading, loadingMessage, onTypingProgress }) => {
  const [animation, setAnimation] = useState(null);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(() => (
    typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
  ));
  const [handledAnimations, setHandledAnimations] = useState(() => new Set());

  const latestMessage = messages[messages.length - 1];
  const latestMessageId = latestMessage ? messageKey(latestMessage, messages.length - 1) : null;
  const newestBotIndex = messages.map((message) => message.sender).lastIndexOf('bot');
  const newestBot = newestBotIndex >= 0 ? messages[newestBotIndex] : null;
  const newestBotId = newestBot ? messageKey(newestBot, newestBotIndex) : null;

  const finishCurrentAnimation = useCallback(() => {
    if (!animation) return;
    setAnimation(null);
  }, [animation]);

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const updatePreference = () => setPrefersReducedMotion(media.matches);
    media.addEventListener?.('change', updatePreference);
    return () => media.removeEventListener?.('change', updatePreference);
  }, []);

  useEffect(() => {
    if (!prefersReducedMotion || !animation) return undefined;
    const finishTimer = window.setTimeout(finishCurrentAnimation, 0);
    return () => window.clearTimeout(finishTimer);
  }, [prefersReducedMotion, animation, finishCurrentAnimation]);

  useEffect(() => {
    if (!animation || latestMessageId === animation.messageId) return undefined;
    const finishTimer = window.setTimeout(finishCurrentAnimation, 0);
    return () => window.clearTimeout(finishTimer);
  }, [latestMessageId, animation, finishCurrentAnimation]);

  useEffect(() => {
    if (!newestBot?.animate || !newestBotId || handledAnimations.has(newestBotId)) return;
    const paragraphs = splitIntoBubbles(newestBot.text);
    const shouldSkipAnimation = newestBot.triage_tag === 'EMERGENCY' || prefersReducedMotion || !paragraphs.length;
    // Animate only after the audited final answer arrives; streaming could expose text that is later rejected or regenerated.
    const startTimer = window.setTimeout(() => {
      setHandledAnimations((current) => {
        if (current.has(newestBotId)) return current;
        const next = new Set(current);
        next.add(newestBotId);
        return next;
      });
      if (!shouldSkipAnimation) {
        setAnimation({ messageId: newestBotId, paragraphs, paragraphIndex: 0, visibleWords: 0, status: 'dots' });
      }
    }, 0);
    return () => window.clearTimeout(startTimer);
  }, [newestBotId, newestBot?.animate, newestBot?.text, newestBot?.triage_tag, prefersReducedMotion, handledAnimations]);

  useEffect(() => {
    if (!animation) return undefined;
    const { paragraphIndex, paragraphs, status, visibleWords } = animation;
    const currentWords = getWords(paragraphs[paragraphIndex] || '');
    let timer;

    if (status === 'dots') {
      const dotsDelay = [700, 430, 350][paragraphIndex] || 350;
      timer = window.setTimeout(() => {
        setAnimation((current) => current?.messageId === animation.messageId
          ? { ...current, status: 'typing' }
          : current);
      }, dotsDelay);
    } else if (visibleWords >= currentWords.length) {
      timer = window.setTimeout(() => {
        if (paragraphIndex + 1 < paragraphs.length) {
          setAnimation((current) => current?.messageId === animation.messageId
            ? { ...current, paragraphIndex: paragraphIndex + 1, visibleWords: 0, status: 'dots' }
            : current);
        } else {
          finishCurrentAnimation();
        }
      }, 120);
    } else {
      const perWordTarget = Math.min(55, 3800 / Math.max(1, currentWords.length));
      const jittered = perWordTarget * (0.9 + Math.random() * 0.2);
      const delay = Math.max(1, Math.min(55, 5000 / Math.max(1, currentWords.length), jittered));
      timer = window.setTimeout(() => {
        setAnimation((current) => current?.messageId === animation.messageId
          ? { ...current, visibleWords: current.visibleWords + 1 }
          : current);
      }, delay);
    }

    return () => window.clearTimeout(timer);
  }, [animation, finishCurrentAnimation]);

  useEffect(() => {
    if (animation) onTypingProgress?.();
  }, [animation, onTypingProgress]);

  useEffect(() => {
    if (!animation) return undefined;
    const onKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        finishCurrentAnimation();
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [animation, finishCurrentAnimation]);

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4" aria-live="polite" aria-relevant="additions text">
      {messages.map((msg, index) => {
        const isUser = msg.sender === 'user';
        const id = messageKey(msg, index);
        const isEmergency = !isUser && msg.triage_tag === 'EMERGENCY';
        const shouldAnimate = !isUser && msg.animate && !isEmergency && !prefersReducedMotion;
        const isAnimating = animation?.messageId === id;
        const hasPlayed = handledAnimations.has(id);
        const metadataVisible = !shouldAnimate || (hasPlayed && !isAnimating);
        const cleanText = isUser ? String(msg.text || '') : normalizeAssistantAnswer(msg.text || '');
        const paragraphs = isUser ? [cleanText] : splitIntoBubbles(cleanText);

        return (
          <div key={id} className={`flex items-start gap-3 animate-fade-in ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
            <div aria-hidden="true" className={`w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 shadow-md ${isUser ? 'bg-gradient-to-tr from-sky-500 to-indigo-500 text-white' : 'bg-gradient-to-tr from-teal-500 to-emerald-500 text-slate-950 font-bold'}`}>
              {isUser ? <User className="w-5 h-5" /> : <Activity className="w-5 h-5" />}
            </div>
            <div className={`flex max-w-[88%] md:max-w-[80%] flex-col gap-2 ${isUser ? 'items-end' : 'items-start'}`}>
              {paragraphs.map((paragraph, paragraphIndex) => {
                if (isAnimating && paragraphIndex > animation.paragraphIndex) return null;
                if (shouldAnimate && !isAnimating && !hasPlayed && paragraphIndex > 0) return null;
                const currentAnimationBubble = isAnimating && paragraphIndex === animation.paragraphIndex;
                const pendingAnimationBubble = shouldAnimate && !isAnimating && !hasPlayed && paragraphIndex === 0;
                const isDots = (currentAnimationBubble && animation.status === 'dots') || pendingAnimationBubble;
                let visibleText = paragraph;
                if (currentAnimationBubble && animation.status === 'typing') {
                  visibleText = getWords(paragraph).slice(0, animation.visibleWords).join('');
                }
                if (currentAnimationBubble && animation.status === 'dots') visibleText = '';

                return (
                  <div
                    key={`${id}-paragraph-${paragraphIndex}`}
                    className={`w-fit max-w-full rounded-2xl p-4 shadow-lg leading-relaxed text-sm ${isUser ? 'bg-gradient-to-r from-teal-600 to-cyan-700 text-white rounded-tr-none' : 'bg-slate-900/90 border border-slate-700/80 text-slate-100 rounded-tl-none'} ${isAnimating ? 'cursor-pointer' : ''}`}
                    onClick={isAnimating ? finishCurrentAnimation : undefined}
                    onKeyDown={isAnimating ? (event) => {
                      if (event.key === 'Enter' || event.key === ' ') finishCurrentAnimation();
                    } : undefined}
                    role={isAnimating ? 'button' : undefined}
                    tabIndex={isAnimating ? 0 : undefined}
                    aria-label={isAnimating ? 'Click to show the complete answer' : undefined}
                  >
                    {paragraphIndex === 0 && (
                      <>
                        <div className="flex items-center justify-between mb-2 text-xs opacity-80">
                          <span className="font-semibold tracking-wide">{isUser ? 'You' : 'SwasthyaSaathi'}</span>
                          {!isUser && metadataVisible && msg.answer_mode && msg.answer_mode !== 'SOCIAL' && (
                            <span className="text-[10px] text-teal-300">{{ GROUNDED: 'From MoHFW/ICMR/WHO guidelines', WEB_TRUSTED: 'From trusted health websites', MIXED: 'Guidelines + general information', GENERAL: 'General health information' }[msg.answer_mode] || ''}</span>
                          )}
                        </div>
                        {isEmergency && <TriageBanner tag={msg.triage_tag} emergencyContact={msg.emergency_contact} />}
                      </>
                    )}
                    <div className="text-sm leading-relaxed">
                      {isDots
                        ? <TypingDots />
                        : isUser ? <div className="whitespace-pre-wrap">{visibleText}</div> : renderAssistantMarkdown(visibleText, `${id}-${paragraphIndex}`)}
                    </div>
                  </div>
                );
              })}

              {metadataVisible && !isUser && !isEmergency && ['URGENT', 'ROUTINE', 'CONSULT_SOON'].includes(msg.triage_tag) && (
                <div className="mt-3"><TriageBanner tag={msg.triage_tag} emergencyContact={msg.emergency_contact} /></div>
              )}
              {metadataVisible && !isUser && msg.tools_used?.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-3" aria-label="Tools used">
                  {msg.tools_used.map((tool) => <span key={tool} className="text-[10px] px-2 py-1 rounded-full bg-slate-800 border border-slate-700 text-slate-300">{tool.replaceAll('_', ' ')}</span>)}
                </div>
              )}
              {metadataVisible && !isUser && (msg.sources?.length > 0 || msg.is_grounded === false) && (
                <SourceCitations sources={msg.sources} isGrounded={msg.is_grounded} />
              )}
              {metadataVisible && !isUser && msg.answer_mode && msg.answer_mode !== 'SOCIAL' && (
                <div className="sr-only">{msg.answer_mode}</div>
              )}
            </div>
          </div>
        );
      })}

      {isLoading && (
        <div role="status" className="flex items-start gap-3 text-slate-300 text-sm">
          <div aria-hidden="true" className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 bg-gradient-to-tr from-teal-500 to-emerald-500 text-slate-950 shadow-md">
            <Activity className="w-5 h-5" />
          </div>
          <div className="rounded-2xl rounded-tl-none p-4 bg-slate-900/90 border border-slate-700/80">
            <TypingDots label={loadingMessage || 'SwasthyaSaathi is typing'} />
            {loadingMessage?.startsWith('Waking up') && <span className="ml-2 text-xs">{loadingMessage}</span>}
          </div>
        </div>
      )}
    </div>
  );
};

export default MessageList;
