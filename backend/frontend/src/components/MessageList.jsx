import React from 'react';
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
            <a href={'tel:' + (emergencyContact || '112')} aria-label="Call emergency services" className="px-3 py-2 rounded-md bg-red-600 hover:bg-red-500 text-white font-bold text-xs flex items-center gap-1">
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

const MessageList = ({ messages, isLoading, loadingMessage }) => (
  <div className="flex-1 overflow-y-auto p-4 space-y-4" aria-live="polite" aria-relevant="additions">
    {messages.map((msg, index) => {
      const isUser = msg.sender === 'user';
      return <div key={`${msg.sender}-${index}`} className={`flex items-start gap-3 animate-fade-in ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        <div aria-hidden="true" className={`w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 shadow-md ${isUser ? 'bg-gradient-to-tr from-sky-500 to-indigo-500 text-white' : 'bg-gradient-to-tr from-teal-500 to-emerald-500 text-slate-950 font-bold'}`}>
          {isUser ? <User className="w-5 h-5" /> : <Activity className="w-5 h-5" />}
        </div>
        <div className={`max-w-[88%] md:max-w-[80%] rounded-2xl p-4 shadow-lg leading-relaxed text-sm ${isUser ? 'bg-gradient-to-r from-teal-600 to-cyan-700 text-white rounded-tr-none' : 'bg-slate-900/90 border border-slate-700/80 text-slate-100 rounded-tl-none'}`}>
          <div className="flex items-center justify-between mb-2 text-xs opacity-80">
            <span className="font-semibold tracking-wide">{isUser ? 'You' : 'SwasthyaSaathi'}</span>
            {!isUser && msg.answer_mode && msg.answer_mode !== 'SOCIAL' && <span className="text-[10px] text-teal-300">{{ GROUNDED: 'From MoHFW/ICMR/WHO guidelines', WEB_TRUSTED: 'From trusted health websites', MIXED: 'Guidelines + general information', GENERAL: 'General health information' }[msg.answer_mode] || ''}</span>}
          </div>
          {!isUser && ['EMERGENCY', 'URGENT', 'ROUTINE'].includes(msg.triage_tag) && <TriageBanner tag={msg.triage_tag} emergencyContact={msg.emergency_contact} />}
          <div className="whitespace-pre-wrap text-sm leading-relaxed">{msg.text}</div>
          {!isUser && msg.sources?.length > 0 && <SourceCitations sources={msg.sources} isGrounded={msg.is_grounded} />}
        </div>
      </div>;
    })}
    {isLoading && <div role="status" className="flex items-center gap-3 p-4 text-slate-300 text-sm"><Activity aria-hidden="true" className="w-5 h-5 text-teal-400 animate-spin" /><span>{loadingMessage || 'Looking for relevant health guidance…'}</span></div>}
  </div>
);

export default MessageList;
