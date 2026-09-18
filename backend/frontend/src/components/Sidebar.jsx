import React from 'react';
import { History, AlertTriangle, CheckCircle, Clock, Stethoscope, ChevronRight, TrendingUp, RefreshCw } from 'lucide-react';

const Sidebar = ({ history = [], pattern = null, onSelectHistoryQuery, isOpen, onClose }) => {
  const getTriageBadge = (tag) => {
    switch (tag) {
      case 'EMERGENCY':
        return (
          <span className="px-2 py-0.5 text-[10px] font-bold rounded bg-red-500/20 text-red-400 border border-red-500/30 flex items-center gap-1">
            <AlertTriangle className="w-3 h-3" /> EMERGENCY
          </span>
        );
      case 'CONSULT_SOON':
        return (
          <span className="px-2 py-0.5 text-[10px] font-semibold rounded bg-amber-500/20 text-amber-300 border border-amber-500/30 flex items-center gap-1">
            <Clock className="w-3 h-3" /> CONSULT SOON
          </span>
        );
      case 'SELF_CARE':
        return (
          <span className="px-2 py-0.5 text-[10px] font-medium rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
            <CheckCircle className="w-3 h-3" /> SELF CARE
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 text-[10px] font-medium rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 flex items-center gap-1">
            <Stethoscope className="w-3 h-3" /> INFO
          </span>
        );
    }
  };

  return (
    <aside className={`fixed inset-y-0 left-0 z-40 w-80 bg-slate-900/95 backdrop-blur-xl border-r border-slate-800 p-4 transition-transform duration-300 flex flex-col md:static md:translate-x-0 ${isOpen ? 'translate-x-0' : '-translate-x-full'}`}>
      <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
        <div className="flex items-center gap-2 text-teal-400 font-semibold text-sm">
          <History className="w-4 h-4 text-teal-300" />
          <span>Longitudinal Health Profile</span>
        </div>
        <button onClick={onClose} className="md:hidden text-slate-400 hover:text-slate-200">
          ✕
        </button>
      </div>

      {/* Pattern Detection Alert Banner */}
      {pattern && pattern.pattern_detected && (
        <div className={`mb-4 p-3 rounded-xl border text-xs shadow-md animate-fade-in ${
          pattern.pattern_type === 'escalating'
            ? 'bg-red-950/80 border-red-500/60 text-red-200'
            : 'bg-amber-950/80 border-amber-500/60 text-amber-200'
        }`}>
          <div className="flex items-center gap-2 font-bold mb-1">
            {pattern.pattern_type === 'escalating' ? (
              <TrendingUp className="w-4 h-4 text-red-400" />
            ) : (
              <RefreshCw className="w-4 h-4 text-amber-400" />
            )}
            <span className="uppercase tracking-wider">
              {pattern.pattern_type} Pattern Alert
            </span>
          </div>
          <p className="text-[11px] leading-relaxed mb-1.5 opacity-90">
            {pattern.description}
          </p>
          <div className="p-2 rounded bg-slate-900/60 border border-slate-800 text-[10px] text-slate-300 font-medium">
            💡 {pattern.recommended_action}
          </div>
        </div>
      )}

      <div className="text-xs text-slate-400 mb-2 font-medium flex items-center justify-between">
        <span>Logged Symptoms (Neon DB)</span>
        <span className="text-[10px] text-teal-400 font-mono">Past 14 Days</span>
      </div>

      <div className="flex-1 overflow-y-auto space-y-2 pr-1">
        {history.length === 0 ? (
          <div className="text-center py-8 text-slate-500 text-xs italic">
            No previous symptom logs recorded yet.
          </div>
        ) : (
          history.map((item) => (
            <div
              key={item.id || Math.random()}
              onClick={() => onSelectHistoryQuery && onSelectHistoryQuery(item.query_text)}
              className="p-2.5 rounded-lg bg-slate-800/60 hover:bg-slate-800 border border-slate-700/50 hover:border-teal-500/50 cursor-pointer transition-all group"
            >
              <div className="flex items-center justify-between mb-1.5">
                {getTriageBadge(item.triage_tag)}
                <span className="text-[10px] text-slate-500 font-mono">
                  {item.created_at ? new Date(item.created_at).toLocaleDateString() : 'Recent'}
                </span>
              </div>
              <p className="text-xs text-slate-200 font-medium line-clamp-2 group-hover:text-teal-300">
                "{item.query_text}"
              </p>
              {item.topic && (
                <div className="mt-1 flex items-center justify-between text-[10px] text-slate-400">
                  <span className="capitalize">Topic: {item.topic.replace('_', ' ')}</span>
                  <ChevronRight className="w-3 h-3 text-slate-600 group-hover:text-teal-400" />
                </div>
              )}
            </div>
          ))
        )}
      </div>

      <div className="mt-auto pt-3 border-t border-slate-800 text-[11px] text-slate-500 flex items-center justify-between">
        <span>PostgreSQL Audit Trail</span>
        <span className="text-emerald-400">Connected</span>
      </div>
    </aside>
  );
};

export default Sidebar;
