import React, { useState } from 'react';
import { BookOpen, ExternalLink, MapPin, ShieldAlert } from 'lucide-react';

const safeUrl = (value) => {
  try {
    const url = new URL(value);
    return ['https:', 'http:'].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
};

function sourceLocation(source) {
  const page = source.page_number ?? source.page ?? source.page_no;
  if (page !== undefined && page !== null && String(page).trim()) return `p. ${page}`;
  return source.section || (source.source_type === 'live_search' ? 'Web source' : '');
}

function uniqueSources(sources) {
  const seen = new Set();
  return (sources || []).filter((source) => {
    const name = source.doc_name || source.title || 'Health guideline';
    const page = source.page_number ?? source.page ?? source.page_no;
    const locationKey = page !== undefined && page !== null ? `page:${page}` : `section:${source.section || ''}`;
    const key = `${name}|${locationKey}|${source.source_url || ''}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

const SourceCitations = ({ sources = [], isGrounded = true }) => {
  const [expandedIdx, setExpandedIdx] = useState(null);
  const citations = uniqueSources(sources);

  if (!citations.length) return (
    <div className="mt-2.5 rounded-xl border border-amber-500/30 bg-amber-950/30 p-3 text-xs text-amber-100" role="note">
      <div className="flex items-start gap-2"><ShieldAlert aria-hidden="true" className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-300" />
        <span>This reply is general information and has no official source citation.</span>
      </div>
    </div>
  );

  const activeSource = expandedIdx === null ? null : citations[expandedIdx];
  const activeTitle = activeSource?.doc_name || activeSource?.title || 'Health guideline';
  const activeUrl = safeUrl(activeSource?.source_url);

  return (
    <section className="mt-1.5 w-full border-t border-slate-700/60 pt-3" aria-label="Sources">
      {isGrounded === false && <div role="note" className="mb-3 rounded-xl border border-amber-500/30 bg-amber-950/30 p-2.5 text-xs text-amber-100">I couldn’t fully match this reply to its references, so please confirm important decisions with a health professional.</div>}
      <div className="flex flex-wrap items-center gap-2">
        <span className="mr-0.5 flex items-center gap-1.5 text-xs font-semibold text-emerald-300"><BookOpen aria-hidden="true" className="h-4 w-4" /> Sources:</span>
        {citations.map((source, idx) => {
          const title = source.doc_name || source.title || 'Health guideline';
          const location = sourceLocation(source);
          const selected = expandedIdx === idx;
          const detailsId = `source-details-${idx}`;
          return (
            <button
              key={`${title}-${location}-${idx}`}
              type="button"
              aria-expanded={selected}
              aria-controls={detailsId}
              title={`${title}${location ? `, ${location}` : ''}`}
              onClick={() => setExpandedIdx(selected ? null : idx)}
              className={`inline-flex max-w-full items-center gap-1.5 rounded-full border px-3 py-1.5 text-left text-xs transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-teal-300 ${selected ? 'border-emerald-300 bg-emerald-900/70 text-emerald-50' : 'border-emerald-500/40 bg-emerald-950/55 text-emerald-100 hover:bg-emerald-900/60'}`}
            >
              <MapPin aria-hidden="true" className="h-3.5 w-3.5 flex-shrink-0 text-emerald-300" />
              <span className="max-w-[min(22rem,65vw)] truncate">{title}{location ? `, ${location}` : ''}</span>
            </button>
          );
        })}
      </div>

      {activeSource && <div id={`source-details-${expandedIdx}`} className="mt-2 rounded-xl border border-emerald-500/30 bg-slate-950/60 p-3 text-xs leading-relaxed text-slate-200">
        <div className="mb-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 font-semibold text-emerald-100">
          <span>{activeTitle}</span>
          {sourceLocation(activeSource) && <span className="font-normal text-slate-300">· {sourceLocation(activeSource)}</span>}
        </div>
        {activeSource.snippet && <p className="whitespace-pre-wrap">{activeSource.snippet}</p>}
        {activeUrl && <a href={activeUrl} target="_blank" rel="noopener noreferrer" className="mt-2 inline-flex items-center gap-1 text-emerald-300 underline underline-offset-2 hover:text-emerald-200">
          Open original source <ExternalLink aria-hidden="true" className="h-3 w-3" /><span className="sr-only"> (opens in a new tab)</span>
        </a>}
      </div>}
    </section>
  );
};

export default SourceCitations;
