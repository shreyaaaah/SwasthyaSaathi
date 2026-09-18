import React, { useState, useEffect } from 'react';
import { Send, Sparkles, Mic, MicOff } from 'lucide-react';

const SYMPTOM_CHIPS = [
  { label: "🤒 High Fever >3 days", query: "What should I do about high fever lasting 3 days?" },
  { label: "🫁 Persistent Cough (3 wks)", query: "I've had a persistent cough for 3 weeks" },
  { label: "🦟 Dengue Signs & Care", query: "What are early warning signs and home care for Dengue fever?" },
  { label: "🫀 Severe Chest Pain", query: "I'm having severe chest pain and can't breathe" },
  { label: "🤰 Antenatal Care", query: "What antenatal checkups and nutrition are needed during pregnancy?" },
  { label: "💊 Antibiotics Myth", query: "Does antibiotics cure viral infections like cold or flu?" },
  { label: "🌿 Turmeric Diabetes Myth", query: "does turmeric cure diabetes" }
];

const MessageInput = ({ onSend, isLoading }) => {
  const [input, setInput] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [speechRecognitionSupported, setSpeechRecognitionSupported] = useState(false);

  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      setSpeechRecognitionSupported(true);
    }
  }, []);

  const startVoiceInput = () => {
    if (!speechRecognitionSupported) {
      alert("Speech recognition is not supported in this browser. Please type your query.");
      return;
    }

    try {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      recognition.lang = 'en-IN'; // Default to Indian English / Hindi mix
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;

      setIsListening(true);
      recognition.start();

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInput(transcript);
        setIsListening(false);
      };

      recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };
    } catch (err) {
      console.error("Voice input error:", err);
      setIsListening(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    onSend(input);
    setInput('');
  };

  const handleChipClick = (query) => {
    if (isLoading) return;
    onSend(query);
  };

  return (
    <div className="p-3 border-t border-slate-700/60 bg-slate-900/80 backdrop-blur-md rounded-b-2xl">
      {/* Quick-Tap Symptom Chips (Low-Literacy Accessibility) */}
      <div className="flex items-center gap-2 overflow-x-auto pb-2.5 scrollbar-none">
        <div className="flex items-center gap-1 text-[11px] font-semibold text-slate-400 whitespace-nowrap">
          <Sparkles className="w-3.5 h-3.5 text-amber-400" />
          <span>Tap Symptom:</span>
        </div>
        {SYMPTOM_CHIPS.map((chip, idx) => (
          <button
            key={idx}
            onClick={() => handleChipClick(chip.query)}
            disabled={isLoading}
            className="text-[11px] px-3 py-1 rounded-full bg-slate-800 hover:bg-teal-900/40 text-slate-200 hover:text-teal-200 border border-slate-700 hover:border-teal-500/50 whitespace-nowrap transition-all duration-150 font-medium"
          >
            {chip.label}
          </button>
        ))}
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        {/* Voice Input Button */}
        <button
          type="button"
          onClick={startVoiceInput}
          disabled={isLoading}
          title={isListening ? "Listening..." : "Click to speak query"}
          className={`p-3 rounded-xl border transition-all ${
            isListening
              ? 'bg-red-500/20 text-red-400 border-red-500 animate-pulse'
              : 'bg-slate-800 hover:bg-slate-700 text-teal-400 border-slate-700 hover:border-teal-500/50'
          }`}
        >
          {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
        </button>

        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={isListening ? "Listening... speak your query now..." : "Ask a health question (e.g. Dengue care, TB symptoms, Pregnancy, Fever)..."}
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
