import React, { useRef, useEffect, useCallback, useState } from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';

const ChatWindow = ({ messages, onSendMessage, isLoading, loadingMessage }) => {
  const bottomRef = useRef(null);
  const [typingProgress, setTypingProgress] = useState(0);
  const reportTypingProgress = useCallback(() => setTypingProgress((value) => value + 1), []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: typingProgress ? 'auto' : 'smooth' });
  }, [messages, isLoading, typingProgress]);

  return (
    <div className="flex-1 flex flex-col glass-card overflow-hidden shadow-2xl my-2 border border-slate-700/60">
      <MessageList messages={messages} isLoading={isLoading} loadingMessage={loadingMessage} onTypingProgress={reportTypingProgress} />
      <div ref={bottomRef} />
      <MessageInput onSend={onSendMessage} isLoading={isLoading} />
    </div>
  );
};

export default ChatWindow;
