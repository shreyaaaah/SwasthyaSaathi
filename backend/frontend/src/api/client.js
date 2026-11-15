import axios from 'axios';

// Use the confirmed production service URL so a stale Vercel env var cannot
// route chat requests to a different or retired Render service.
const API_BASE_URL = import.meta.env.DEV
  ? (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000')
  : 'https://swasthyasaathi-3.onrender.com';

let activeChatController = null;

function extractDisplayText(value) {
  if (typeof value === 'string') return value.trim();
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (Array.isArray(value)) {
    return value.map(extractDisplayText).filter(Boolean).join('\n');
  }
  if (value && typeof value === 'object') {
    for (const key of ['answer', 'message', 'detail', 'text', 'content', 'output', 'response', 'msg']) {
      const text = extractDisplayText(value[key]);
      if (text) return text;
    }
  }
  return '';
}

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const sendChatMessage = async (query, conversationHistory = []) => {
  activeChatController?.abort();
  const controller = new AbortController();
  activeChatController = controller;
  try {
    const response = await apiClient.post('/chat', {
      query,
      user_id: 'anonymous',
      conversation_history: conversationHistory,
    }, { signal: controller.signal });
    const data = response.data && typeof response.data === 'object' ? response.data : {};
    const answer = extractDisplayText(data.answer ?? data.response ?? data.message);
    return {
      ...data,
      answer: answer || 'Sorry, I couldn’t format the reply just now. Please try sending your question again.',
    };
  } catch (error) {
    console.error('API Error sending chat message:', error);
    if (axios.isCancel(error)) throw error;
    if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
      throw new Error('The health service is taking too long to respond. Please try again shortly.');
    }
    const responseMessage = extractDisplayText(error.response?.data?.detail)
      || extractDisplayText(error.response?.data?.message);
    if (responseMessage) {
      throw new Error(responseMessage);
    }
    throw new Error('Unable to connect to the health information service. Please try again shortly.');
  } finally {
    if (activeChatController === controller) activeChatController = null;
  }
};

export const fetchHealthStatus = async () => {
  try {
    const response = await apiClient.get('/health', { timeout: 8000 });
    return response.data;
  } catch {
    return { status: 'offline', api_available: false, vector_store_ready: false, embedding_model_ready: false, source_corpus_ready: false, unindexed_documents: [], retrieval_ready: false, vectors_count: 0 };
  }
};

export default apiClient;
