import React, { useState } from 'react';
import { Send, Sparkles, Mic, MicOff } from 'lucide-react';

const SYMPTOM_CHIPS = [
  { label: 'High fever', query: 'What should I do about high fever lasting 3 days?' },
  { label: 'Persistent cough', query: "I've had a persistent cough for 3 weeks" },
  { label: 'Dengue signs and care', query: 'What are early warning signs and home care for Dengue fever?' },
  { label: 'Severe chest pain', query: "I'm having severe chest pain and can't breathe" },
  { label: 'Pregnancy care', query: 'What antenatal checkups and nutrition are needed during pregnancy?' },
  { label: 'Antibiotics and viruses', query: 'Do antibiotics cure viral infections like cold or flu?' },
  { label: 'Turmeric and diabetes', query: 'Does turmeric cure diabetes?' }
];

const MessageInput = ({ onSend, isLoading }) => {
  const [input, setInput] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [voiceError, setVoiceError] = useState('');
  const voiceSupported = typeof window !== 'undefined' && ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window);

  const startVoiceInput = () => {
    setVoiceError('');
    if (!voiceSupported) {
      setVoiceError('Voice input is not supported in this browser. Please type your question.');
      return;
    }
    try {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      recognition.lang = 'en-IN';
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;
      setIsListening(true);
      recognition.start();
      recognition.onresult = (event) => setInput(event.results[0][0].transcript);
      recognition.onerror = () => setVoiceError('Voice input could not start. You can type your question instead.');
      recognition.onend = () => setIsListening(false);
    } catch {
      setIsListening(false);
      setVoiceError('Voice input could not start. You can type your question instead.');
    }
  };

  const send = (text) => {
    if (!text.trim() || isLoading) return;
    onSend(text.trim());
    setInput('');
  };

  return <div className="p-3 border-t border-slate-700/60 bg-slate-900/80 backdrop-blur-md rounded-b-2xl">
    <div className="flex items-center gap-2 overflow-x-auto pb-2.5" aria-label="Suggested questions">
      <span className="flex items-center gap-1 text-[11px] font-semibold text-slate-300 whitespace-nowrap"><Sparkles aria-hidden="true" className="w-3.5 h-3.5 text-amber-400" /> Try a topic:</span>
      {SYMPTOM_CHIPS.map((chip) => <button key={chip.label} type="button" onClick={() => send(chip.query)} disabled={isLoading} className="text-[11px] px-3 py-1.5 rounded-full bg-slate-800 hover:bg-teal-900/40 text-slate-100 border border-slate-600 hover:border-teal-400 whitespace-nowrap disabled:opacity-50">{chip.label}</button>)}
    </div>
    <form onSubmit={(event) => { event.preventDefault(); send(input); }} className="flex items-center gap-2">
      <button type="button" onClick={startVoiceInput} disabled={isLoading || isListening} aria-label="Start voice input" title="Start voice input" className="p-3 rounded-xl border bg-slate-800 hover:bg-slate-700 text-teal-300 border-slate-600 disabled:opacity-50">
        {isListening ? <MicOff aria-hidden="true" className="w-4 h-4" /> : <Mic aria-hidden="true" className="w-4 h-4" />}
      </button>
      <input type="text" value={input} onChange={(event) => setInput(event.target.value)} placeholder={isListening ? 'Listening… speak now' : 'Ask a health question…'} disabled={isLoading} aria-label="Ask a health question" className="min-w-0 flex-1 bg-slate-950/80 border border-slate-600 focus:border-teal-400 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500/30" />
      <button type="submit" disabled={!input.trim() || isLoading} className="px-4 py-3 rounded-xl bg-gradient-to-r from-teal-400 to-emerald-400 hover:from-teal-300 hover:to-emerald-300 text-slate-950 font-bold text-sm flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed">
        <span>Send</span><Send aria-hidden="true" className="w-4 h-4" />
      </button>
    </form>
    <p className="mt-2 text-[11px] text-slate-300">Voice input may be processed by your browser’s speech recognition service. Messages are not saved to a personal history.</p>
    {voiceError && <p role="status" className="mt-1 text-xs text-amber-200">{voiceError}</p>}
  </div>;
};

export default MessageInput;
