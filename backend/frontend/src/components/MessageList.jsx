import React from 'react';
import { User, Activity, AlertTriangle, Clock, CheckCircle, Info, PhoneCall, Cpu, Wrench } from 'lucide-react';
import SourceCitations from './SourceCitations';

const MessageList = ({ messages, isLoading }) => {
  const renderTriageBanner = (triageTag) => {
    switch (triageTag) {
      case 'EMERGENCY':
        return (
          <div className="mb-3 p-3 rounded-xl bg-red-950/80 border-2 border-red-500 text-red-200 shadow-lg shadow-red-500/20 animate-pulse">
            <div className="flex items-start gap-2.5">
              <div className="p-1.5 rounded-lg bg-red-600 text-white flex-shrink-0">
                <AlertTriangle className="w-5 h-5" />
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-bold text-red-300 tracking-wide">
                    🚨 MEDICAL EMERGENCY DETECTED
                  </h4>
                  <a
                    href="tel:112"
                    className="px-2.5 py-1 rounded-md bg-red-600 hover:bg-red-500 text-white font-bold text-xs flex items-center gap-1 shadow transition-colors"
                  >
                    <PhoneCall className="w-3.5 h-3.5" /> Call 112 / 108
                  </a>
                </div>
                <p className="text-xs text-red-200 mt-1">
                  Immediate emergency clinical evaluation required. Do not delay seeking medical care.
                </p>
              </div>
            </div>
          </div>
        );

      case 'CONSULT_SOON':
        return (
          <div className="mb-3 p-2.5 rounded-xl bg-amber-950/60 border border-amber-500/50 text-amber-200">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-amber-400 flex-shrink-0" />
              <div>
                <span className="text-xs font-bold text-amber-300">
                  ⚠️ Doctor Consultation Recommended (within 24-48 hrs)
                </span>
              </div>
            </div>
          </div>
        );

      case 'SELF_CARE':
        return (
          <div className="mb-3 p-2.5 rounded-xl bg-emerald-950/60 border border-emerald-500/40 text-emerald-200">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
              <div>
                <span className="text-xs font-semibold text-emerald-300">
                  🌿 Home Care & Symptom Monitoring
                </span>
              </div>
            </div>
          </div>
        );

      case 'GENERAL_INFO':
        return (
          <div className="mb-3 p-2 rounded-xl bg-cyan-950/40 border border-cyan-500/30 text-cyan-200 text-xs flex items-center gap-2">
            <Info className="w-4 h-4 text-cyan-400 flex-shrink-0" />
            <span>Public Health & Guideline Information</span>
          </div>
        );

      default:
        return null;
    }
  };

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

            {/* Message Card */}
            <div
              className={`max-w-[88%] md:max-w-[80%] rounded-2xl p-4 shadow-lg leading-relaxed text-sm ${
                isUser
                  ? 'bg-gradient-to-r from-teal-600 to-cyan-700 text-white rounded-tr-none'
                  : 'bg-slate-900/90 border border-slate-700/80 text-slate-100 rounded-tl-none'
              }`}
            >
              {/* Header Info */}
              <div className="flex items-center justify-between mb-2 text-xs opacity-80">
                <span className="font-semibold tracking-wide">
                  {isUser ? 'You' : 'SwasthyaSaathi AI Agent'}
                </span>

                {!isUser && msg.tools_used && msg.tools_used.length > 0 && (
                  <div className="flex items-center gap-1 text-[10px] text-teal-300 font-mono bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700">
                    <Wrench className="w-3 h-3 text-teal-400" />
                    <span>Tools: {msg.tools_used.join(', ')}</span>
                  </div>
                )}
              </div>

              {/* Urgency Triage Banner */}
              {!isUser && msg.triage_tag && renderTriageBanner(msg.triage_tag)}

              {/* Message Content */}
              <div className="whitespace-pre-wrap text-sm leading-relaxed font-sans">
                {msg.text}
              </div>

              {/* Source Citations */}
              {!isUser && msg.sources && msg.sources.length > 0 && (
                <SourceCitations sources={msg.sources} />
              )}
            </div>
          </div>
        );
      })}

      {/* Loading Indicator */}
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
            <span className="text-xs text-teal-300 font-medium">
              Orchestrator selecting tools & searching MoHFW guidelines...
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default MessageList;
