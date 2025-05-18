import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ExpandableContainer from './ExpandableContainer';
import { TrashIcon } from './Icons'; // Import TrashIcon
import { API_URL } from '../config';

function CodebaseManager({
  codebases,
  selectedCodebase,
  setSelectedCodebase,
  newCodeText,
  setNewCodeText,
  newCodeDescription,
  setNewCodeDescription,
  newCodebaseName,
  setNewCodebaseName,
  selectedBaseCodebase,
  setSelectedBaseCodebase,
  handleCreateCodebase,
  handleAddCode,
  handleSelectCodebase, // This is actually deleteCodebase in App.jsx, will rename for clarity if needed
  fetchCodebases
}) {
  const [activeCodebase, setActiveCodebase] = useState(null);
  const [codebaseContent, setCodebaseContent] = useState([]);
  const [loading, setLoading] = useState(false);
  
  // Fetch codebase content when a codebase is selected or codebases list is refreshed
  useEffect(() => {
    if (activeCodebase) {
      fetchCodebaseContent(activeCodebase);
    }
  }, [activeCodebase, codebases]); // Added codebases to dependency array based on original logic
  
  const fetchCodebaseContent = async (codebaseName) => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/codebases/${codebaseName}`);
      setCodebaseContent(response.data.codes || []);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching codebase content:', error);
      setLoading(false);
    }
  };
  
  // Handle selection of a codebase (for viewing/editing)
  const handleCodebaseSelectInternal = (codebaseName) => {
    setActiveCodebase(codebaseName);
    setSelectedCodebase(codebaseName); // This prop updates config in FileConfigForm
  };
  
  return (
    <div className="codebases-panel">
      <h3>Codebases Management</h3>
      
      <div className="codebases-layout">
        <div className="codebases-list-panel">
          <div className="codebases-section">
            <h4>Available Codebases</h4>
            <div className="codebases-list">
              <h5>Default Codebases</h5>
              <ul>
                {codebases.default_codebases?.map(codebase => (
                  <li 
                    key={codebase.filename}
                    className={activeCodebase === codebase.filename ? 'active' : ''}
                    onClick={() => handleCodebaseSelectInternal(codebase.filename)}
                  >
                    {codebase.filename} <span className="code-count">({codebase.code_count} codes)</span>
                  </li>
                ))}
                {(!codebases.default_codebases || codebases.default_codebases.length === 0) && (
                  <li className="empty-message">No default codebases</li>
                )}
              </ul>
              
              <h5>User Codebases</h5>
              <ul>
                {codebases.user_codebases?.map(codebase => (
                  <li 
                    key={codebase.filename}
                    className={activeCodebase === codebase.filename ? 'active' : ''}
                    onClick={() => handleCodebaseSelectInternal(codebase.filename)}
                  >
                    {codebase.filename} <span className="code-count">({codebase.code_count} codes)</span>
                    <button 
                      className="delete-button"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (window.confirm(`Are you sure you want to delete ${codebase.filename}?`)) {
                          handleSelectCodebase(codebase.filename); // This is the delete function passed from App.jsx via FileConfigForm
                          // fetchCodebases(); // fetchCodebases is called within the passed delete function in App.jsx
                        }
                      }}
                    >
                      <TrashIcon />
                    </button>
                  </li>
                ))}
                {(!codebases.user_codebases || codebases.user_codebases.length === 0) && (
                  <li className="empty-message">No user codebases</li>
                )}
              </ul>
            </div>
          </div>
          
          <div className="codebases-section">
            <h4>Create New Codebase</h4>
            <div className="create-codebase-form">
              <div className="form-group">
                <label>Name:</label>
                <input 
                  type="text" 
                  value={newCodebaseName}
                  onChange={(e) => setNewCodebaseName(e.target.value)}
                  placeholder="Enter codebase name"
                />
              </div>
              
              <div className="form-group">
                <label>Base on existing:</label>
                <select 
                  value={selectedBaseCodebase}
                  onChange={(e) => setSelectedBaseCodebase(e.target.value)}
                >
                  <option value="">-- None (create empty) --</option>
                  {codebases.default_codebases?.map(codebase => (
                    <option key={codebase.filename} value={codebase.filename}>
                      {codebase.filename}
                    </option>
                  ))}
                  {codebases.user_codebases?.map(codebase => (
                    <option key={codebase.filename} value={codebase.filename}>
                      {codebase.filename}
                    </option>
                  ))}
                </select>
              </div>
              
              <button 
                className="primary-button" 
                onClick={handleCreateCodebase}
                disabled={!newCodebaseName}
              >
                Create Codebase
              </button>
            </div>
          </div>
        </div>
        
        <div className="codebase-content-panel">
          <div className="codebases-section">
            <h4>
              {activeCodebase ? `Codes in ${activeCodebase}` : 'Select a codebase'}
            </h4>
            
            {loading ? (
              <div className="loading">Loading codes...</div>
            ) : (
              <>
                {activeCodebase && (
                  <>
                    <ExpandableContainer title={`Codes in ${activeCodebase}`}>
                      <div className="codes-list">
                        {codebaseContent.length > 0 ? (
                          <table className="codes-table">
                            <thead>
                              <tr>
                                <th>Code</th>
                                <th>Description</th>
                              </tr>
                            </thead>
                            <tbody>
                              {codebaseContent.map((code, index) => (
                                <tr key={index}>
                                  <td>{code.text}</td>
                                  <td>{code.metadata?.description || '-'}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        ) : (
                          <div className="empty-message">No codes in this codebase</div>
                        )}
                      </div>
                    </ExpandableContainer>
                    
                    {/* Form to add new code */}
                    <div className="add-code-form">
                      <h4>Add New Code</h4>
                      <div className="form-group">
                        <label>Code text:</label>
                        <input 
                          type="text" 
                          value={newCodeText}
                          onChange={(e) => setNewCodeText(e.target.value)}
                          placeholder="Enter code text"
                        />
                      </div>
                      
                      <div className="form-group">
                        <label>Description:</label>
                        <textarea 
                          value={newCodeDescription}
                          onChange={(e) => setNewCodeDescription(e.target.value)}
                          placeholder="Enter code description"
                          rows={3}
                        />
                      </div>
                      
                      <button 
                        className="primary-button" 
                        onClick={handleAddCode}
                        disabled={!newCodeText}
                      >
                        Add Code
                      </button>
                    </div>
                  </>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default CodebaseManager; 