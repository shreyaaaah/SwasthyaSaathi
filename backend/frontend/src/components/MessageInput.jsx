import React, { useState } from 'react';
import { Send, Sparkles } from 'lucide-react';

const SUGGESTED_QUESTIONS = [
  "What are early warning signs of Dengue?",
  "What is the standard treatment regimen for Tuberculosis?",
  "What vaccines are scheduled at 6 weeks for infants?",
  "What antipyretic is safe for dengue fever?"
];

const MessageInput = ({ onSend, isLoading }) => {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    onSend(input);
    setInput('');
  };

  const handleChipClick = (question) => {
    if (isLoading) return;
    onSend(question);
  };

  return (
    <div className="p-3 border-t border-slate-700/60 bg-slate-900/80 backdrop-blur-md rounded-b-2xl">
      {/* Sample Question Chips */}
      <div className="flex items-center gap-2 overflow-x-auto pb-2.5 scrollbar-none">
        <div className="flex items-center gap-1 text-[11px] font-semibold text-slate-400 whitespace-nowrap">
          <Sparkles className="w-3 h-3 text-amber-400" />
          <span>Quick queries:</span>
        </div>
        {SUGGESTED_QUESTIONS.map((q, idx) => (
          <button
            key={idx}
            onClick={() => handleChipClick(q)}
            disabled={isLoading}
            className="text-[11px] px-2.5 py-1 rounded-full bg-slate-800 hover:bg-teal-900/40 text-slate-300 hover:text-teal-200 border border-slate-700 hover:border-teal-500/40 whitespace-nowrap transition-all duration-150"
          >
            {q}
          </button>
        ))}
      </div>

      {/* Input Box */}
      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a public health question (e.g., Dengue symptoms, TB care, Maternal health)..."
          disabled={isLoading}
          className="flex-1 bg-slate-950/80 border border-slate-700 focus:border-teal-500 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/20 transition-all"
        />
        <button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="px-5 py-3 rounded-xl bg-gradient-to-r from-teal-500 to-emerald-500 hover:from-teal-600 hover:to-emerald-600 text-slate-950 font-bold text-sm flex items-center gap-2 shadow-lg shadow-teal-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          <span>Send</span>
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
};

export default MessageInput;
