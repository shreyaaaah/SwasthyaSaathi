import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const sendChatMessage = async (query, userId = 'user_001') => {
  try {
    const response = await apiClient.post('/chat', {
      query,
      user_id: userId,
    });
    return response.data;
  } catch (error) {
    console.error('API Error sending chat message:', error);
    if (error.response && error.response.data && error.response.data.detail) {
      throw new Error(error.response.data.detail);
    }
    throw new Error('Unable to connect to SwasthyaSaathi backend. Please ensure the server is running on port 8000.');
  }
};

export const fetchHealthStatus = async () => {
  try {
    const response = await apiClient.get('/health');
    return response.data;
  } catch (error) {
    return { status: 'offline', vector_store_ready: false, vectors_count: 0 };
  }
};

export const fetchUserHistory = async (userId = 'user_001') => {
  try {
    const response = await apiClient.get(`/api/history/${userId}`);
    return response.data;
  } catch (error) {
    console.error('API Error fetching user history:', error);
    return [];
  }
};

export default apiClient;
