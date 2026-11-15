import React, { useEffect, useState } from 'react';
import Header from './components/Header';
import ChatWindow from './components/ChatWindow';
import BmiCalculator from './components/BmiCalculator';
import QuickHelpPanel from './components/QuickHelpPanel';
import HospitalFinder from './components/HospitalFinder';
import { sendChatMessage, fetchHealthStatus } from './api/client';
import { Calculator, ChevronDown, HeartPulse, MapPin } from 'lucide-react';

const INITIAL_WELCOME_MESSAGE = {
  id: 'welcome',
  sender: 'bot',
  text: 'Hey! 😊 How are you doing today? If something’s on your mind, tell me about it—I’m here to help.',
  sources: [],
  is_grounded: true,
  triage_tag: null,
  answer_mode: 'SOCIAL',
  tools_used: []
};

const GREETING_PATTERN = /^(?:hi|hii+|hello|hey(?: there)?|namaste|namaskar)[!.?\s]*$/i;

function createMessageId() {
  return typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : `message-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function App() {
  const [messages, setMessages] = useState([INITIAL_WELCOME_MESSAGE]);
  const [isLoading, setIsLoading] = useState(false);
  const [isWakingUp, setIsWakingUp] = useState(false);
  const [healthStatus, setHealthStatus] = useState(null);

  useEffect(() => {
    fetchHealthStatus().then(setHealthStatus);
  }, []);

  const handleSendMessage = async (userQuery) => {
    if (GREETING_PATTERN.test(userQuery.trim())) {
      setMessages((prev) => [...prev,
        { id: createMessageId(), sender: 'user', text: userQuery },
        {
          id: createMessageId(),
          sender: 'bot',
          text: 'Hey! 😊 How are you doing? What’s on your mind today? 💚',
          sources: [],
          is_grounded: null,
          triage_tag: null,
          answer_mode: 'SOCIAL',
          tools_used: [],
          animate: true
        }
      ]);
      return;
    }

    setMessages((prev) => [...prev, { id: createMessageId(), sender: 'user', text: userQuery }]);
    setIsLoading(true);
    setIsWakingUp(false);
    const wakeTimer = window.setTimeout(() => setIsWakingUp(true), 8000);
    try {
      const conversationHistory = messages.slice(-6).map((message) => ({
        role: message.sender === 'user' ? 'user' : 'assistant',
        // The backend limits each history turn to 2,000 chars. Keep a margin
        // so long assistant replies cannot make the next chat request fail.
        content: String(message.text ?? '').slice(0, 1900)
      }));
      const response = await sendChatMessage(userQuery, conversationHistory);
      setMessages((prev) => [...prev, {
        id: createMessageId(),
        sender: 'bot',
        text: response.answer,
        review_query: userQuery,
        sources: response.sources || [],
        is_grounded: response.is_grounded,
        triage_tag: response.triage_tag || 'GENERAL_INFO',
        tools_used: response.tools_used || [],
        answer_mode: response.answer_mode,
        gate_scores: response.gate_scores,
        emergency_contact: response.emergency_contact,
        animate: true
      }]);
    } catch (error) {
      if (error?.code === 'ERR_CANCELED' || error?.name === 'AbortError') return;
      const errorText = typeof error?.message === 'string' && error.message !== '[object Object]'
        ? error.message
        : 'We could not get a readable response. Please try again in a moment.';
      setMessages((prev) => [...prev, {
        id: createMessageId(),
        sender: 'bot',
        text: errorText,
        sources: [],
        is_grounded: false,
        triage_tag: 'ERROR',
        tools_used: []
      }]);
    } finally {
      window.clearTimeout(wakeTimer);
      setIsLoading(false);
      setIsWakingUp(false);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 text-slate-100 font-sans">
      <aside aria-label="Health tools" className="hidden w-[300px] shrink-0 flex-col gap-4 overflow-y-auto border-r border-slate-800 bg-slate-900/40 p-4 xl:flex">
        <p className="px-1 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Health tools</p>
        <details className="group">
          <summary className="flex min-h-12 cursor-pointer list-none items-center justify-between rounded-xl border border-slate-700 bg-slate-900/80 p-3 text-sm font-semibold text-slate-100 [&::-webkit-details-marker]:hidden">
            <span className="flex items-center gap-2"><Calculator aria-hidden="true" className="h-4 w-4 text-teal-300" />BMI calculator</span>
            <ChevronDown aria-hidden="true" className="h-4 w-4 text-slate-400 transition-transform group-open:rotate-180" />
          </summary>
          <div className="pt-3">
            <BmiCalculator showTitle={false} />
            <p className="px-1 pt-3 text-[11px] leading-relaxed text-slate-500">For adults 20 and older. Under 20 needs age- and sex-specific BMI charts.</p>
          </div>
        </details>
        <details className="group">
          <summary className="flex min-h-12 cursor-pointer list-none items-center justify-between rounded-xl border border-slate-700 bg-slate-900/80 p-3 text-sm font-semibold text-slate-100 [&::-webkit-details-marker]:hidden">
            <span className="flex items-center gap-2"><HeartPulse aria-hidden="true" className="h-4 w-4 text-teal-300" />Quick Help</span>
            <ChevronDown aria-hidden="true" className="h-4 w-4 text-slate-400 transition-transform group-open:rotate-180" />
          </summary>
          <div className="pt-3"><QuickHelpPanel onAsk={handleSendMessage} isLoading={isLoading} compact /></div>
        </details>
      </aside>
      <main className="flex min-w-0 flex-1 flex-col overflow-hidden px-3 py-2">
        <details className="group mb-2 rounded-xl border border-slate-700 bg-slate-900/80 p-3 xl:hidden">
          <summary className="flex cursor-pointer list-none items-center justify-between text-sm font-semibold text-teal-200 [&::-webkit-details-marker]:hidden">
            <span className="flex items-center gap-2"><Calculator aria-hidden="true" className="h-4 w-4" />BMI calculator</span>
            <ChevronDown aria-hidden="true" className="h-4 w-4 text-slate-400 transition-transform group-open:rotate-180" />
          </summary>
          <div className="pt-3">
            <BmiCalculator showTitle={false} />
            <p className="px-1 pt-3 text-[11px] leading-relaxed text-slate-500">For adults 20 and older. Under 20 needs age- and sex-specific BMI charts.</p>
          </div>
        </details>
        <Header status={healthStatus} />
        <details className="group mt-1 rounded-xl border border-slate-700 bg-slate-900/80 p-3 xl:hidden">
          <summary className="flex cursor-pointer list-none items-center justify-between text-sm font-semibold text-teal-200 [&::-webkit-details-marker]:hidden">
            <span className="flex items-center gap-2"><HeartPulse aria-hidden="true" className="h-4 w-4" />Quick Help</span>
            <ChevronDown aria-hidden="true" className="h-4 w-4 text-slate-400 transition-transform group-open:rotate-180" />
          </summary>
          <div className="pt-3"><QuickHelpPanel onAsk={handleSendMessage} isLoading={isLoading} compact /></div>
        </details>
        <details className="group mt-1 rounded-xl border border-slate-700 bg-slate-900/80 p-3 2xl:hidden">
          <summary className="flex cursor-pointer list-none items-center justify-between text-sm font-semibold text-teal-200 [&::-webkit-details-marker]:hidden">
            <span className="flex items-center gap-2"><MapPin aria-hidden="true" className="h-4 w-4" />Nearby hospitals</span>
            <ChevronDown aria-hidden="true" className="h-4 w-4 text-slate-400 transition-transform group-open:rotate-180" />
          </summary>
          <div className="pt-3"><HospitalFinder compact /></div>
        </details>
        <ChatWindow messages={messages} onSendMessage={handleSendMessage} isLoading={isLoading} loadingMessage={isWakingUp ? 'Waking up the server, this can take a minute…' : 'Looking for relevant health guidance…'} />
        <details className="group mx-auto w-full max-w-3xl px-2 py-1 text-center">
          <summary className="inline-flex cursor-pointer list-none items-center gap-1 text-[11px] text-slate-500 hover:text-slate-300 [&::-webkit-details-marker]:hidden">
            Privacy &amp; data use
            <ChevronDown aria-hidden="true" className="h-3 w-3 transition-transform group-open:rotate-180" />
          </summary>
          <p className="mx-auto max-w-3xl pt-2 text-[11px] leading-relaxed text-slate-500">
            Health information only. Full chats are not saved. For answers using trusted websites or general information, a normalized topic is retained to help improve the guidance library. Your question and up to six recent turns may go to the AI service; web searches may also be sent to a search provider and health websites. Avoid sharing identifying details.
          </p>
        </details>
      </main>
      <aside aria-label="Nearby hospitals" className="hidden w-[290px] shrink-0 flex-col gap-4 overflow-y-auto border-l border-slate-800 bg-slate-900/40 p-4 2xl:flex">
        <details className="group">
          <summary className="flex min-h-12 cursor-pointer list-none items-center justify-between rounded-xl border border-slate-700 bg-slate-900/80 p-3 text-sm font-semibold text-slate-100 [&::-webkit-details-marker]:hidden">
            <span className="flex items-center gap-2"><MapPin aria-hidden="true" className="h-4 w-4 text-teal-300" />Nearby hospitals</span>
            <ChevronDown aria-hidden="true" className="h-4 w-4 text-slate-400 transition-transform group-open:rotate-180" />
          </summary>
          <div className="pt-3"><HospitalFinder compact /></div>
        </details>
      </aside>
    </div>
  );
}

export default App;
