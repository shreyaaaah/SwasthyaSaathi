import React from 'react';
import { AlertTriangle, ArrowUpRight, HeartPulse, PhoneCall, ShieldCheck } from 'lucide-react';

const QUICK_QUESTIONS = [
  { label: 'Fever care', query: 'I have a fever. What can I do, and when should I seek medical care?' },
  { label: 'Pregnancy concern', query: 'I have a pregnancy concern. Which warning signs mean I should seek urgent care?' },
  { label: 'Dengue guidance', query: 'What are dengue warning signs, and what should I do?' }
];

function QuickHelpPanel({ onAsk, isLoading = false, compact = false }) {
  return (
    <section aria-label="Quick health help" className={`space-y-4 ${compact ? '' : 'rounded-2xl border border-slate-700/80 bg-slate-900/70 p-4 shadow-xl'}`}>
      {!compact && (
        <div>
          <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
            <HeartPulse aria-hidden="true" className="h-4 w-4 text-teal-300" />
            Quick Help
          </h2>
          <p className="mt-1 text-xs leading-relaxed text-slate-400">Trusted next steps for urgent concerns and common questions.</p>
        </div>
      )}

      <div className="rounded-xl border border-rose-900/70 bg-rose-950/30 p-3">
        <div className="flex items-start gap-2">
          <AlertTriangle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-rose-300" />
          <div>
            <h3 className="text-xs font-semibold text-rose-100">If someone may be in immediate danger</h3>
            <p className="mt-1 text-[11px] leading-relaxed text-slate-300">For emergency help in India, call 112 or go to the nearest emergency department.</p>
          </div>
        </div>
        <a href="tel:112" className="mt-3 flex min-h-10 items-center justify-center gap-2 rounded-lg bg-rose-600 px-3 py-2 text-sm font-semibold text-white transition hover:bg-rose-500 focus:outline-none focus:ring-2 focus:ring-rose-300">
          <PhoneCall aria-hidden="true" className="h-4 w-4" />
          Call 112
        </a>
      </div>

      <details className="group rounded-xl border border-slate-700 bg-slate-950/40 p-3">
        <summary className="flex cursor-pointer list-none items-center gap-2 text-xs font-semibold text-slate-200">
          <ShieldCheck aria-hidden="true" className="h-4 w-4 text-teal-300" />
          When to get urgent help
          <span aria-hidden="true" className="ml-auto text-slate-500 transition group-open:rotate-180">⌄</span>
        </summary>
        <ul className="mt-3 list-disc space-y-2 pl-4 text-[11px] leading-relaxed text-slate-300 marker:text-teal-300">
          <li>Severe chest pain or serious difficulty breathing.</li>
          <li>Sudden face drooping, one-sided weakness, or trouble speaking.</li>
          <li>Heavy bleeding, fainting, or seizures.</li>
          <li>During pregnancy: bleeding, severe abdominal pain, fits, or waters breaking—contact maternity care or a hospital promptly.</li>
        </ul>
        <p className="mt-3 text-[10px] leading-relaxed text-slate-500">If symptoms seem life-threatening, call 112. Don’t wait for a chatbot reply.</p>
      </details>

      <div>
        <h3 className="mb-2 text-xs font-semibold text-slate-200">Ask about</h3>
        <div className="grid grid-cols-1 gap-2">
          {QUICK_QUESTIONS.map((item) => (
            <button
              key={item.label}
              type="button"
              onClick={() => onAsk(item.query)}
              disabled={isLoading}
              className="flex min-h-9 items-center justify-between gap-2 rounded-lg border border-slate-700 bg-slate-800/70 px-3 py-2 text-left text-[11px] text-slate-200 transition hover:border-teal-500/70 hover:bg-teal-950/40 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {item.label}
              <ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5 shrink-0 text-teal-300" />
            </button>
          ))}
        </div>
      </div>

      <div className="border-t border-slate-800 pt-3">
        <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-500">Trusted guidance</p>
        <a className="flex items-center justify-between gap-2 py-1 text-[11px] text-teal-200 hover:text-teal-100" href="https://www.mha.gov.in/emergency-response-support-system-erss" target="_blank" rel="noreferrer">
          India emergency number 112 <ArrowUpRight aria-hidden="true" className="h-3 w-3 shrink-0" />
        </a>
        <a className="flex items-center justify-between gap-2 py-1 text-[11px] text-teal-200 hover:text-teal-100" href="https://nhm.gov.in/New-Update-2022-24/CH-Programmes/HBNC-%26-HBYC-Resource-%20Material/Revised-India-MCP-Card-English-%2828-05-2018%29.pdf" target="_blank" rel="noreferrer">
          Pregnancy danger signs <ArrowUpRight aria-hidden="true" className="h-3 w-3 shrink-0" />
        </a>
      </div>
    </section>
  );
}

export default QuickHelpPanel;
