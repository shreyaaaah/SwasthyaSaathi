import React from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';

const ChatWindow = ({ messages, onSendMessage, isLoading, loadingMessage }) => {
  return (
    <div className="flex-1 flex flex-col glass-card overflow-hidden shadow-2xl my-2 border border-slate-700/60">
      <MessageList messages={messages} isLoading={isLoading} loadingMessage={loadingMessage} />
      <MessageInput onSend={onSendMessage} isLoading={isLoading} />
    </div>
  );
};

export default ChatWindow;
