import React, { useEffect, useState } from 'react';
import Header from './components/Header';
import ChatWindow from './components/ChatWindow';
import { sendChatMessage, fetchHealthStatus } from './api/client';

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
        content: message.text
      }));
      const response = await sendChatMessage(userQuery, conversationHistory);
      setMessages((prev) => [...prev, {
        id: createMessageId(),
        sender: 'bot',
        text: response.answer,
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
      setMessages((prev) => [...prev, {
        id: createMessageId(),
        sender: 'bot',
        text: error.message || 'We could not get a response. Please try again in a moment.',
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
      <main className="flex-1 flex flex-col h-full overflow-hidden px-3 py-2 max-w-6xl mx-auto w-full">
        <Header status={healthStatus} />
        <ChatWindow messages={messages} onSendMessage={handleSendMessage} isLoading={isLoading} loadingMessage={isWakingUp ? 'Waking up the server, this can take a minute…' : 'Looking for relevant health guidance…'} />
        <p className="px-2 py-2 text-center text-[11px] text-slate-400">
          Health information only. Full chats are not saved. For answers using trusted websites or general information, a normalized topic is retained to help improve the guidance library. Your question and up to six recent turns may go to the AI service; web searches may also be sent to a search provider and health websites. Avoid sharing identifying details.
        </p>
      </main>
    </div>
  );
}

export default App;
