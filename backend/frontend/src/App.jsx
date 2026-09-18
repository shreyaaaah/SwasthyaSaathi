import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import ChatWindow from './components/ChatWindow';
import { sendChatMessage, fetchHealthStatus } from './api/client';

const INITIAL_WELCOME_MESSAGE = {
  sender: 'bot',
  text: `Namaste! I am SwasthyaSaathi (स्वास्थ्य साथी), your Public Health AI assistant grounded in official guidelines from the Ministry of Health and Family Welfare (MoHFW), ICMR, NHP, and WHO.\n\nHow can I assist you with public health information today? You can ask about dengue symptoms, tuberculosis treatment guidelines, maternal care, or child immunization.`,
  sources: [],
  is_grounded: true,
  triage_tag: 'WELCOME'
};

function App() {
  const [messages, setMessages] = useState([INITIAL_WELCOME_MESSAGE]);
  const [isLoading, setIsLoading] = useState(false);
  const [healthStatus, setHealthStatus] = useState(null);

  useEffect(() => {
    const checkHealth = async () => {
      const status = await fetchHealthStatus();
      setHealthStatus(status);
    };
    checkHealth();
  }, []);

  const handleSendMessage = async (userQuery) => {
    const userMsg = { sender: 'user', text: userQuery };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const response = await sendChatMessage(userQuery);
      
      const botMsg = {
        sender: 'bot',
        text: response.answer,
        sources: response.sources || [],
        is_grounded: response.is_grounded,
        triage_tag: response.triage_tag || 'GENERAL_HEALTH_INFO'
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch (error) {
      const errorMsg = {
        sender: 'bot',
        text: `⚠️ ${error.message || 'An error occurred while fetching information.'}`,
        sources: [],
        is_grounded: false,
        triage_tag: 'ERROR'
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full px-4 py-2 max-w-5xl mx-auto w-full">
      <Header status={healthStatus} />
      <ChatWindow
        messages={messages}
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
      />
    </div>
  );
}

export default App;
