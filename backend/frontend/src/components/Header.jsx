import React from 'react';
import { Activity, ShieldCheck, Database } from 'lucide-react';

const Header = ({ status }) => {
  const apiOnline = status?.status !== 'offline' && Boolean(status);
  const indexReady = Boolean(status?.vector_store_ready);

  return (
    <header className="glass-card my-3 p-4 flex flex-col md:flex-row items-center justify-between gap-4 border-b border-slate-700/50">
      <div className="flex items-center gap-3">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-tr from-teal-500 to-emerald-400 flex items-center justify-center shadow-lg shadow-teal-500/20">
          <Activity className="w-7 h-7 text-slate-950 font-bold" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold bg-gradient-to-r from-teal-300 via-emerald-200 to-cyan-300 bg-clip-text text-transparent">
              SwasthyaSaathi
            </h1>
            <span className="text-xs px-2 py-0.5 rounded-full bg-teal-500/20 text-teal-300 border border-teal-500/30 font-mono">
              Health information
            </span>
          </div>
          <p className="text-xs text-slate-400 flex items-center gap-1.5 mt-0.5">
            <ShieldCheck className="w-3.5 h-3.5 text-teal-400" />
            Public health guidance from MoHFW, ICMR, NHP and WHO
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-center gap-3 text-xs">
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700">
          <Database className="w-3.5 h-3.5 text-emerald-400" />
          <span className="text-slate-300">FAISS Index:</span>
          <span className="font-semibold text-emerald-300 font-mono">
            {indexReady ? `${status.vectors_count || 0} chunks` : 'Unavailable'}
          </span>
        </div>

        <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700">
          <span className={`w-2 h-2 rounded-full ${apiOnline ? 'bg-emerald-400' : 'bg-red-400'}`}></span>
          <span className="text-slate-300 font-medium">
            {statusLabel(status)}
          </span>
        </div>
      </div>
    </header>
  );
};

function statusLabel(status) {
  if (!status) return 'Checking service';
  if (status.status === 'offline') return 'Service offline';
  return status.vector_store_ready ? 'API online · guidelines available' : 'API online · guidelines unavailable';
}

export default Header;
