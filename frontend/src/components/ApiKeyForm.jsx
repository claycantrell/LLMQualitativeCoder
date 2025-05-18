import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_URL } from '../config';

function ApiKeyForm() {
  const [apiKey, setApiKey] = useState('');
  const [isKeySet, setIsKeySet] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [message, setMessage] = useState('');
  const [showKey, setShowKey] = useState(false);

  useEffect(() => {
    // Check if API key is already set
    checkApiKey();
  }, []);

  const checkApiKey = async () => {
    try {
      const response = await axios.get(`${API_URL}/check-api-key`);
      setIsKeySet(response.data.api_key_set);
      if (response.data.api_key_set) {
        setMessage('API key is set');
      }
    } catch (error) {
      console.error('Error checking API key:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSaving(true);
    setMessage('');

    try {
      const response = await axios.post(`${API_URL}/set-api-key`, { api_key: apiKey });
      setIsKeySet(true);
      setMessage(response.data.message);
      setApiKey(''); // Clear the input for security
    } catch (error) {
      setMessage(`Error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="api-key-form">
      <h3>OpenAI API Key</h3>
      {isKeySet ? (
        <div className="key-status success">
          <p>{message}</p>
          <button 
            className="secondary-button"
            onClick={() => {
              setIsKeySet(false);
              setMessage('');
            }}
          >
            Update Key
          </button>
        </div>
      ) : (
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>
              Enter your OpenAI API Key:
              <div className="api-key-input-container">
                <input
                  type={showKey ? "text" : "password"}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  className="api-key-input"
                  placeholder="sk-..."
                  required
                />
                <button
                  type="button"
                  className="toggle-key-visibility"
                  onClick={() => setShowKey(!showKey)}
                >
                  {showKey ? "Hide" : "Show"}
                </button>
              </div>
            </label>
            <small>Your API key is stored in memory and will be cleared when the server restarts.</small>
          </div>
          
          {message && <div className="message">{message}</div>}
          
          <div className="form-actions">
            <button 
              type="submit" 
              className="primary-button" 
              disabled={isSaving || !apiKey}
            >
              {isSaving ? 'Saving...' : 'Save API Key'}
            </button>
          </div>
        </form>
      )}
    </div>
  );
}

export default ApiKeyForm; 