import React, { useState } from 'react';
import { BookOpen, ChevronDown, ChevronUp, FileText, ShieldAlert, ExternalLink } from 'lucide-react';

const authorityFor = (url = '', title = '') => {
  const value = `${url} ${title}`.toLowerCase();
  if (value.includes('who.int') || value.includes('world health organization')) return 'World Health Organization';
  if (value.includes('mohfw') || value.includes('mohfw.gov.in')) return 'Ministry of Health and Family Welfare (India)';
  if (value.includes('icmr')) return 'Indian Council of Medical Research';
  if (value.includes('nhp.gov.in') || value.includes('national health portal')) return 'National Health Portal (India)';
  if (value.includes('cdc.gov')) return 'U.S. Centers for Disease Control and Prevention';
  return '';
};

const safeUrl = (value) => {
  try {
    const url = new URL(value);
    return ['https:', 'http:'].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
};

const SourceCitations = ({ sources = [], isGrounded = true }) => {
  const [expandedIdx, setExpandedIdx] = useState(null);
  const groundedSources = sources || [];

  if (!groundedSources.length) return (
    <div className="mt-3 pt-2.5 border-t border-slate-700/60" role="note">
      <div className="p-2.5 rounded-xl bg-amber-950/40 border border-amber-500/40 text-amber-100 text-xs flex items-start gap-2.5">
        <ShieldAlert aria-hidden="true" className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
        <div><strong className="block text-amber-200">General information; no official guideline match</strong>
          <span className="block mt-1">This response could not be matched to an official guideline. It may be incomplete; ask a qualified health professional for advice about your situation.</span>
        </div>
      </div>
    </div>
  );

  return <section className="mt-3 pt-3 border-t border-slate-700/60" aria-label="Sources">
    {isGrounded === false && <div role="note" className="mb-3 p-2.5 rounded-xl bg-amber-950/40 border border-amber-500/40 text-amber-100 text-xs">The answer could not be fully matched to its references. Treat it as general information and confirm important decisions with a qualified health professional.</div>}
    <div className="flex items-center gap-1.5 text-xs font-semibold text-teal-300 mb-2"><BookOpen aria-hidden="true" className="w-3.5 h-3.5" /><span>References ({groundedSources.length})</span></div>
    <div className="flex flex-wrap gap-2">
      {groundedSources.map((source, idx) => {
        const expanded = expandedIdx === idx;
        const title = source.doc_name || source.title || 'Health guideline';
        const url = safeUrl(source.source_url);
        const authority = authorityFor(source.source_url, title);
        const live = source.source_type === 'live_search';
        return <article key={`${title}-${idx}`} className={`w-full text-xs rounded-lg border overflow-hidden ${live ? 'border-cyan-500/40 bg-cyan-950/20' : 'border-teal-500/30 bg-teal-950/20'}`}>
          <button type="button" aria-expanded={expanded} onClick={() => setExpandedIdx(expanded ? null : idx)} className="w-full px-3 py-2 flex items-center justify-between gap-3 text-left hover:bg-slate-800/40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-teal-400">
            <span className="flex items-center gap-2 min-w-0"><FileText aria-hidden="true" className="w-3.5 h-3.5 text-teal-300 flex-shrink-0" /><span className="min-w-0"><span className="block font-medium text-slate-100 truncate">{title}</span><span className="block text-slate-300 truncate">{authority || source.section || (live ? 'Live search result' : 'Guideline reference')}</span></span></span>
            <span className="flex items-center gap-2 flex-shrink-0 text-slate-300">{live ? 'Live source' : ''}{expanded ? <ChevronUp aria-hidden="true" className="w-4 h-4" /> : <ChevronDown aria-hidden="true" className="w-4 h-4" />}</span>
          </button>
          {expanded && <div className="px-3 py-2.5 bg-slate-900/70 border-t border-slate-800 text-slate-200 leading-relaxed">
            {source.section && <p className="mb-2"><strong>Section:</strong> {source.section}</p>}
            <p className="whitespace-pre-wrap">{source.snippet || 'No excerpt is available for this reference.'}</p>
            {url && <a href={url} target="_blank" rel="noopener noreferrer" className="mt-2 inline-flex items-center gap-1 text-cyan-300 underline underline-offset-2">Open original source <ExternalLink aria-hidden="true" className="w-3 h-3" /><span className="sr-only"> (opens in a new tab)</span></a>}
          </div>}
        </article>;
      })}
    </div>
  </section>;
};

export default SourceCitations;
