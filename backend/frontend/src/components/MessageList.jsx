import React from 'react';
import { User, Activity, AlertCircle, Info, Tag } from 'lucide-react';
import SourceCitations from './SourceCitations';

const MessageList = ({ messages, isLoading }) => {
  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((msg, index) => {
        const isUser = msg.sender === 'user';
        
        return (
          <div
            key={index}
            className={`flex items-start gap-3 animate-fade-in ${
              isUser ? 'flex-row-reverse' : 'flex-row'
            }`}
          >
            {/* Avatar */}
            <div
              className={`w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 shadow-md ${
                isUser
                  ? 'bg-gradient-to-tr from-sky-500 to-indigo-500 text-white'
                  : 'bg-gradient-to-tr from-teal-500 to-emerald-500 text-slate-950 font-bold'
              }`}
            >
              {isUser ? <User className="w-5 h-5" /> : <Activity className="w-5 h-5" />}
            </div>

            {/* Bubble Content */}
            <div
              className={`max-w-[85%] md:max-w-[75%] rounded-2xl p-4 shadow-lg leading-relaxed text-sm ${
                isUser
                  ? 'bg-gradient-to-r from-teal-600 to-cyan-700 text-white rounded-tr-none'
                  : 'bg-slate-900/90 border border-slate-700/80 text-slate-100 rounded-tl-none'
              }`}
            >
              {/* Sender Label & Triage Tag */}
              <div className="flex items-center justify-between mb-1.5 text-xs opacity-75">
                <span className="font-semibold tracking-wide">
                  {isUser ? 'You' : 'SwasthyaSaathi AI'}
                </span>

                {!isUser && msg.triage_tag && (
                  <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-teal-500/20 text-teal-300 border border-teal-500/30">
                    <Tag className="w-3 h-3" />
                    {msg.triage_tag}
                  </span>
                )}
              </div>

              {/* Message Text */}
              <div className="whitespace-pre-wrap text-sm leading-relaxed">
                {msg.text}
              </div>

              {/* Grounded Status Banner */}
              {!isUser && msg.is_grounded === false && (
                <div className="mt-2.5 p-2 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  <span>Notice: No direct match in official health documents.</span>
                </div>
              )}

              {/* Source Citations */}
              {!isUser && msg.sources && msg.sources.length > 0 && (
                <SourceCitations sources={msg.sources} />
              )}
            </div>
          </div>
        );
      })}

      {/* Loading indicator */}
      {isLoading && (
        <div className="flex items-start gap-3 animate-fade-in">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-teal-500 to-emerald-500 text-slate-950 flex items-center justify-center shadow-md">
            <Activity className="w-5 h-5 animate-spin" />
          </div>
          <div className="bg-slate-900/90 border border-slate-700 p-4 rounded-2xl rounded-tl-none text-slate-300 text-sm flex items-center gap-3">
            <div className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-teal-400 animate-bounce" style={{ animationDelay: '0ms' }}></span>
              <span className="w-2 h-2 rounded-full bg-teal-400 animate-bounce" style={{ animationDelay: '150ms' }}></span>
              <span className="w-2 h-2 rounded-full bg-teal-400 animate-bounce" style={{ animationDelay: '300ms' }}></span>
            </div>
            <span className="text-xs text-teal-300">Searching FAISS index & generating grounded answer...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default MessageList;
