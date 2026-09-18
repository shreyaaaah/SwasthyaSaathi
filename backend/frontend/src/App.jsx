import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import ChatWindow from './components/ChatWindow';
import Sidebar from './components/Sidebar';
import { sendChatMessage, fetchHealthStatus, fetchUserHistory } from './api/client';
import { Menu } from 'lucide-react';

const INITIAL_WELCOME_MESSAGE = {
  sender: 'bot',
  text: `Namaste! I am SwasthyaSaathi (स्वास्थ्य साथी), your AI Public Health Assistant grounded in official guidelines from MoHFW, ICMR, NHP, and WHO.

I am powered by an agentic tool-calling orchestrator with live FAISS retrieval, medical myth detection, session history logs, and triage urgency evaluation.

How can I assist you today? You can tap any of the quick symptom chips below or type your question.`,
  sources: [],
  is_grounded: true,
  triage_tag: 'GENERAL_INFO',
  tools_used: []
};

const USER_ID = 'test_user_001';

function App() {
  const [messages, setMessages] = useState([INITIAL_WELCOME_MESSAGE]);
  const [isLoading, setIsLoading] = useState(false);
  const [healthStatus, setHealthStatus] = useState(null);
  const [userHistory, setUserHistory] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const loadHistory = async () => {
    const historyData = await fetchUserHistory(USER_ID);
    setUserHistory(historyData);
  };

  useEffect(() => {
    const checkHealth = async () => {
      const status = await fetchHealthStatus();
      setHealthStatus(status);
    };
    checkHealth();
    loadHistory();
  }, []);

  const handleSendMessage = async (userQuery) => {
    const userMsg = { sender: 'user', text: userQuery };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const response = await sendChatMessage(userQuery, USER_ID);
      
      const botMsg = {
        sender: 'bot',
        text: response.answer,
        sources: response.sources || [],
        is_grounded: response.is_grounded,
        triage_tag: response.triage_tag || 'GENERAL_INFO',
        tools_used: response.tools_used || []
      };

      setMessages((prev) => [...prev, botMsg]);

      // Refresh sidebar history from Neon DB
      loadHistory();
    } catch (error) {
      const errorMsg = {
        sender: 'bot',
        text: `⚠️ ${error.message || 'An error occurred while fetching information.'}`,
        sources: [],
        is_grounded: false,
        triage_tag: 'ERROR',
        tools_used: []
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectHistoryQuery = (queryText) => {
    setSidebarOpen(false);
    handleSendMessage(queryText);
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 text-slate-100 font-sans">
      {/* Sidebar for Symptom History (Neon DB) */}
      <Sidebar
        history={userHistory}
        onSelectHistoryQuery={handleSelectHistoryQuery}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      {/* Main Container */}
      <div className="flex-1 flex flex-col h-full overflow-hidden px-3 py-2 max-w-6xl mx-auto w-full">
        {/* Top bar on mobile */}
        <div className="flex items-center gap-2 md:hidden mb-1">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg bg-slate-900 border border-slate-700 text-teal-400"
          >
            <Menu className="w-5 h-5" />
          </button>
          <span className="text-xs font-bold text-teal-300">SwasthyaSaathi Agent</span>
        </div>

        <Header status={healthStatus} />

        <ChatWindow
          messages={messages}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
}

export default App;
