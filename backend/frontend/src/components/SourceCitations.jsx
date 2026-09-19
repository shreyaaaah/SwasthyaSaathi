import React, { useState } from 'react';
import { BookOpen, ChevronDown, ChevronUp, FileText, CheckCircle2, ShieldAlert } from 'lucide-react';

const SourceCitations = ({ sources, isGrounded = true }) => {
  const [expandedIdx, setExpandedIdx] = useState(null);

  const hasSources = sources && sources.length > 0;

  if (!hasSources || isGrounded === false) {
    return (
      <div className="mt-3 pt-2.5 border-t border-slate-700/60">
        <div className="p-2.5 rounded-xl bg-amber-950/40 border border-amber-500/40 text-amber-200 text-xs flex items-start gap-2.5 shadow-sm">
          <ShieldAlert className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-bold text-amber-300 block">
              ℹ️ General AI Response (No Official Guideline Match)
            </span>
            <span className="text-[11px] text-amber-200/90 leading-tight block mt-0.5">
              No official MoHFW, ICMR, or WHO guideline matched this question with high confidence (&ge;45%). This answer is provided for general information only — please consult a qualified doctor for clinical guidance.
            </span>
          </div>
        </div>
      </div>
    );
  }

  const toggleExpand = (idx) => {
    setExpandedIdx(expandedIdx === idx ? null : idx);
  };

  return (
    <div className="mt-3 pt-3 border-t border-slate-700/60">
      <div className="flex items-center gap-1.5 text-xs font-semibold text-teal-400 mb-2">
        <BookOpen className="w-3.5 h-3.5" />
        <span>Grounded Sources ({sources.length}):</span>
      </div>

      <div className="flex flex-wrap gap-2">
        {sources.map((source, idx) => {
          const isExpanded = expandedIdx === idx;
          const isLiveSearch = source.source_type === "live_search";

          const cardStyle = isLiveSearch 
            ? "border-cyan-500/40 bg-cyan-950/20 hover:bg-cyan-900/20"
            : "border-teal-500/30 bg-teal-950/20 hover:bg-teal-900/20";
          const iconColor = isLiveSearch ? "text-cyan-400" : "text-teal-400";
          const badgeStyle = isLiveSearch 
            ? "bg-cyan-500/20 text-cyan-300 border border-cyan-400/30"
            : "bg-emerald-500/20 text-emerald-300";

          return (
            <div 
              key={idx} 
              className={`w-full text-xs rounded-lg border overflow-hidden transition-all duration-200 ${cardStyle}`}
            >
              <button
                onClick={() => toggleExpand(idx)}
                className="w-full px-3 py-2 flex items-center justify-between text-left transition-colors"
              >
                <div className="flex items-center gap-2 truncate pr-2">
                  <FileText className={`w-3.5 h-3.5 ${iconColor} flex-shrink-0`} />
                  <span className="font-medium text-slate-200 truncate">
                    {source.doc_name || source.title}
                  </span>
                  <span className="text-slate-400 text-[11px] truncate">
                    ({source.section || "Guideline"})
                  </span>
                </div>

                <div className="flex items-center gap-2 flex-shrink-0">
                  {isLiveSearch ? (
                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold ${badgeStyle}`}>
                      🔍 Live Search
                    </span>
                  ) : source.score ? (
                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${badgeStyle}`}>
                      {(source.score * 100).toFixed(0)}% match
                    </span>
                  ) : null}
                  {isExpanded ? (
                    <ChevronUp className="w-3.5 h-3.5 text-slate-400" />
                  ) : (
                    <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
                  )}
                </div>
              </button>

              {isExpanded && (
                <div className="px-3 py-2.5 bg-slate-900/60 border-t border-slate-800 text-slate-300 text-[11px] leading-relaxed animate-fade-in">
                  <div className="flex items-center justify-between gap-1 text-teal-300 font-semibold mb-1">
                    <div className="flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3 text-teal-400" />
                      <span>Reference Excerpt:</span>
                    </div>
                    {source.source_url && (
                      <a 
                        href={source.source_url} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="text-cyan-400 hover:underline text-[10px] flex items-center gap-1"
                      >
                        🔗 Open Source Page
                      </a>
                    )}
                  </div>
                  <p className="italic text-slate-300 bg-slate-950/50 p-2 rounded border border-slate-800 max-h-48 overflow-y-auto">
                    "{source.snippet || source.content_snippet}"
                  </p>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default SourceCitations;
