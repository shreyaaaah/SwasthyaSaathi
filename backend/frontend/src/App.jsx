import React, { useEffect, useState } from 'react';
import Header from './components/Header';
import ChatWindow from './components/ChatWindow';
import { sendChatMessage, fetchHealthStatus } from './api/client';

const INITIAL_WELCOME_MESSAGE = {
  sender: 'bot',
  text: 'Namaste! What health question can I help with?',
  sources: [],
  is_grounded: true,
  triage_tag: null,
  answer_mode: 'SOCIAL',
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
      const conversationHistory = messages.slice(-6).map((message) => ({
        role: message.sender === 'user' ? 'user' : 'assistant',
        content: message.text
      }));
      const response = await sendChatMessage(userQuery, conversationHistory);
      setMessages((prev) => [...prev, {
        sender: 'bot',
        text: response.answer,
        sources: response.sources || [],
        is_grounded: response.is_grounded,
        triage_tag: response.triage_tag || 'GENERAL_INFO',
        tools_used: response.tools_used || [],
        answer_mode: response.answer_mode,
        gate_scores: response.gate_scores,
        emergency_contact: response.emergency_contact
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
          Health information only. Full chats are not saved. For answers using trusted websites or general information, a normalized topic is retained to help improve the guidance library. Your question and up to six recent turns may go to the AI service; web searches may also be sent to a search provider and health websites. Avoid sharing identifying details.
        </p>
      </main>
    </div>
  );
}

export default App;
