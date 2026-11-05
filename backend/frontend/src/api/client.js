import axios from 'axios';

// Keep production usable even if the Vercel environment variable was omitted.
// Local development still targets the local FastAPI server.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (
  import.meta.env.DEV
    ? 'http://localhost:8000'
    : 'https://swasthya-saathi-3.onrender.com'
);

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const sendChatMessage = async (query, conversationHistory = []) => {
  try {
    const response = await apiClient.post('/chat', {
      query,
      user_id: 'anonymous',
      conversation_history: conversationHistory,
    });
    return response.data;
  } catch (error) {
    console.error('API Error sending chat message:', error);
    if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
      throw new Error('The health service is taking too long to respond. Please try again shortly.');
    }
    if (error.response && error.response.data && error.response.data.detail) {
      throw new Error(error.response.data.detail);
    }
    throw new Error('Unable to connect to the health information service. Please try again shortly.');
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
