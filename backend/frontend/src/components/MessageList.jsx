import React from 'react';
import { User, Activity, AlertTriangle, Clock, CheckCircle, Info, PhoneCall, Wrench } from 'lucide-react';
import SourceCitations from './SourceCitations';

function TriageBanner({ tag }) {
  if (tag === 'EMERGENCY') return (
    <div role="alert" className="mb-3 p-3 rounded-xl bg-red-950/80 border-2 border-red-500 text-red-100">
      <div className="flex items-start gap-2.5">
        <AlertTriangle aria-hidden="true" className="w-5 h-5 text-red-300 flex-shrink-0" />
        <div className="flex-1">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h4 className="text-sm font-bold">Possible emergency</h4>
            <a href="tel:112" aria-label="Call India's emergency number 112" className="px-3 py-2 rounded-md bg-red-600 hover:bg-red-500 text-white font-bold text-xs flex items-center gap-1">
              <PhoneCall aria-hidden="true" className="w-4 h-4" /> Call 112
            </a>
          </div>
          <p className="text-xs mt-1">If you may be in immediate danger, call India’s emergency number or go to the nearest emergency department now. This assistant cannot assess emergencies.</p>
        </div>
      </div>
    </div>
  );
  if (tag === 'CONSULT_SOON') return <div className="mb-3 p-2.5 rounded-xl bg-amber-950/60 border border-amber-500/50 text-amber-200 flex items-center gap-2"><Clock aria-hidden="true" className="w-4 h-4 text-amber-400" /><span className="text-xs font-bold">Medical advice recommended soon</span></div>;
  if (tag === 'SELF_CARE') return <div className="mb-3 p-2.5 rounded-xl bg-emerald-950/60 border border-emerald-500/40 text-emerald-200 flex items-center gap-2"><CheckCircle aria-hidden="true" className="w-4 h-4 text-emerald-400" /><span className="text-xs font-semibold">Home care and symptom monitoring</span></div>;
  if (tag === 'GENERAL_INFO') return <div className="mb-3 p-2 rounded-xl bg-cyan-950/40 border border-cyan-500/30 text-cyan-100 text-xs flex items-center gap-2"><Info aria-hidden="true" className="w-4 h-4 text-cyan-400" /><span>Public health information</span></div>;
  return null;
}

const MessageList = ({ messages, isLoading }) => (
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
            {!isUser && msg.tools_used?.length > 0 && <span className="flex items-center gap-1 text-[10px] text-teal-300"><Wrench aria-hidden="true" className="w-3 h-3" /> Sources checked</span>}
          </div>
          {!isUser && msg.triage_tag && <TriageBanner tag={msg.triage_tag} />}
          <div className="whitespace-pre-wrap text-sm leading-relaxed">{msg.text}</div>
          {!isUser && <SourceCitations sources={msg.sources} isGrounded={msg.is_grounded} />}
        </div>
      </div>;
    })}
    {isLoading && <div role="status" className="flex items-center gap-3 p-4 text-slate-300 text-sm"><Activity aria-hidden="true" className="w-5 h-5 text-teal-400 animate-spin" /><span>Looking for relevant health guidance…</span></div>}
  </div>
);

export default MessageList;
