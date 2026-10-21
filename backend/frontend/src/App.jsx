import React, { useEffect, useState } from 'react';
import Header from './components/Header';
import ChatWindow from './components/ChatWindow';
import { sendChatMessage, fetchHealthStatus } from './api/client';

const INITIAL_WELCOME_MESSAGE = {
  sender: 'bot',
  text: `Namaste! I can help you find general health information from public health guidance. Tell me what you would like to know, or choose a prompt below.\n\nI cannot diagnose or replace a clinician. If someone may be in immediate danger, contact local emergency services now.`,
  sources: [],
  is_grounded: true,
  triage_tag: 'GENERAL_INFO',
  tools_used: []
};

function App() {
  const [messages, setMessages] = useState([INITIAL_WELCOME_MESSAGE]);
  const [isLoading, setIsLoading] = useState(false);
  const [healthStatus, setHealthStatus] = useState(null);

  useEffect(() => {
    fetchHealthStatus().then(setHealthStatus);
  }, []);

  const handleSendMessage = async (userQuery) => {
    setMessages((prev) => [...prev, { sender: 'user', text: userQuery }]);
    setIsLoading(true);
    try {
      const response = await sendChatMessage(userQuery);
      setMessages((prev) => [...prev, {
        sender: 'bot',
        text: response.answer,
        sources: response.sources || [],
        is_grounded: response.is_grounded,
        triage_tag: response.triage_tag || 'GENERAL_INFO',
        tools_used: response.tools_used || []
      }]);
    } catch (error) {
      setMessages((prev) => [...prev, {
        sender: 'bot',
        text: error.message || 'We could not get a response. Please try again in a moment.',
        sources: [],
        is_grounded: false,
        triage_tag: 'ERROR',
        tools_used: []
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 text-slate-100 font-sans">
      <main className="flex-1 flex flex-col h-full overflow-hidden px-3 py-2 max-w-6xl mx-auto w-full">
        <Header status={healthStatus} />
        <ChatWindow messages={messages} onSendMessage={handleSendMessage} isLoading={isLoading} />
        <p className="px-2 py-2 text-center text-[11px] text-slate-400">
          Health information only. Chats are not saved as personal history, but are sent to service providers to generate answers. Avoid sharing names or other identifying details.
        </p>
      </main>
    </div>
  );
}

export default App;
