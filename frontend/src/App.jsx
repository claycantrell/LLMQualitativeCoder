import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import './App.css'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faFileCode, faTrash, faUpload, faFileAlt, faDownload } from '@fortawesome/free-solid-svg-icons'

// API base URL - change this to your API URL when deployed
const API_URL = 'http://localhost:8000'

// Icons as SVG components
const DownloadIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
    <polyline points="7 10 12 15 17 10"></polyline>
    <line x1="12" y1="15" x2="12" y2="3"></line>
  </svg>
)

const DocumentIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
    <polyline points="14 2 14 8 20 8"></polyline>
    <line x1="16" y1="13" x2="8" y2="13"></line>
    <line x1="16" y1="17" x2="8" y2="17"></line>
    <polyline points="10 9 9 9 8 9"></polyline>
  </svg>
)

const TrashIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6"></polyline>
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
    <line x1="10" y1="11" x2="10" y2="17"></line>
    <line x1="14" y1="11" x2="14" y2="17"></line>
  </svg>
)

const UploadIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
    <polyline points="17 8 12 3 7 8"></polyline>
    <line x1="12" y1="3" x2="12" y2="15"></line>
  </svg>
)

// File Configuration Component
function FileConfigForm({ file, onSubmit, onCancel }) {
  const [analyzing, setAnalyzing] = useState(true);
  const [fileStructure, setFileStructure] = useState(null);
  const [error, setError] = useState(null);
  
  // Add active tab state
  const [activeTab, setActiveTab] = useState('config'); // 'config', 'inductive', 'deductive', 'codebases'
  
  // Configuration state
  const [config, setConfig] = useState({
    file_id: file.filename,
    content_field: '',
    context_fields: [],
    list_field: '',
    coding_mode: 'inductive',
    use_parsing: true,
    preliminary_segments_per_prompt: 5,
    meaning_units_per_assignment_prompt: 10,
    context_size: 5,
    model_name: 'gpt-4o-mini',
    temperature: 0.7,
    max_tokens: 2000,
    thread_count: 2,
    selected_codebase: ''
  });
  
  // Add states for prompt editing
  const [inductivePrompt, setInductivePrompt] = useState({ content: '', isCustom: false, loading: true });
  const [deductivePrompt, setDeductivePrompt] = useState({ content: '', isCustom: false, loading: true });
  const [promptSaving, setPromptSaving] = useState(false);
  const [promptError, setPromptError] = useState(null);
  
  // Add states for codebase management
  const [codebases, setCodebases] = useState({ default_codebases: [], user_codebases: [], loading: true });
  const [selectedCodebase, setSelectedCodebase] = useState(null);
  const [codebaseContents, setCodebaseContents] = useState({ codes: [], is_default: false, loading: true });
  const [newCodeText, setNewCodeText] = useState('');
  const [newCodeDescription, setNewCodeDescription] = useState('');
  const [newCodebaseName, setNewCodebaseName] = useState('');
  const [baseCodebase, setBaseCodebase] = useState('');
  const [codebaseError, setCodebaseError] = useState(null);
  const [saving, setSaving] = useState(false);
  
  // Analyze file structure on component mount
  useEffect(() => {
    const analyzeFile = async () => {
      try {
        setAnalyzing(true);
        setError(null);
        
        console.log(`Analyzing file: ${file.filename}`);
        console.log(`URL: ${API_URL}/analyze-file/${file.filename}`);
        
        const response = await axios.get(`${API_URL}/analyze-file/${file.filename}`);
        
        console.log('Analysis response:', response.data);
        
        // Update file structure and suggested mappings
        setFileStructure(response.data.structure);
        
        // Update config with suggested mappings
        const suggestedMappings = response.data.structure.suggested_mappings;
        setConfig(prev => ({
          ...prev,
          content_field: suggestedMappings.content_field || '',
          context_fields: suggestedMappings.context_fields || [],
          list_field: suggestedMappings.list_field || ''
        }));
        
      } catch (err) {
        console.error('Error analyzing file:', err);
        console.error('Error details:', err.response?.data || err.message);
        setError(err.response?.data?.detail || 'Failed to analyze file structure');
      } finally {
        setAnalyzing(false);
      }
    };
    
    analyzeFile();
  }, [file.filename]);
  
  // Handle form field changes
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setConfig({
      ...config,
      [name]: type === 'checkbox' ? checked : 
              type === 'number' ? Number(value) :
              value
    });
  };
  
  // Handle context field selection
  const handleContextFieldChange = (fieldName, isChecked) => {
    setConfig(prev => {
      const updatedContextFields = isChecked
        ? [...prev.context_fields, fieldName]
        : prev.context_fields.filter(f => f !== fieldName);
      
      return {
        ...prev,
        context_fields: updatedContextFields
      };
    });
  };
  
  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    
    // If it's a default file and we're using known formats, we can use the simpler endpoint
    if (file.is_default && (fileStructure?.is_known_format || !config.content_field)) {
      // Use the simpler API endpoint for default files with known formats
      const simpleConfig = {
        coding_mode: config.coding_mode,
        use_parsing: config.use_parsing,
        preliminary_segments_per_prompt: config.preliminary_segments_per_prompt,
        meaning_units_per_assignment_prompt: config.meaning_units_per_assignment_prompt,
        context_size: config.context_size,
        model_name: config.model_name,
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        thread_count: config.thread_count,
        input_file: file.filename
      };
      
      onSubmit(simpleConfig, true); // true indicates this is a standard config
    } else {
      // Use the dynamic configuration endpoint
      onSubmit(config, false); // false indicates this is a dynamic config
    }
  };
  
  // Add functions to fetch and save prompts
  const fetchPrompt = async (promptType) => {
    try {
      const response = await axios.get(`${API_URL}/prompts/${promptType}`);
      if (promptType === 'inductive') {
        setInductivePrompt({
          content: response.data.content,
          isCustom: response.data.is_custom,
          loading: false
        });
      } else {
        setDeductivePrompt({
          content: response.data.content,
          isCustom: response.data.is_custom,
          loading: false
        });
      }
      setPromptError(null);
    } catch (err) {
      console.error(`Error fetching ${promptType} prompt:`, err);
      if (promptType === 'inductive') {
        setInductivePrompt(prev => ({ ...prev, loading: false }));
      } else {
        setDeductivePrompt(prev => ({ ...prev, loading: false }));
      }
      setPromptError(`Failed to load ${promptType} prompt: ${err.response?.data?.detail || err.message}`);
    }
  };

  const savePrompt = async (promptType, content) => {
    setPromptSaving(true);
    setPromptError(null);
    try {
      const response = await axios.post(`${API_URL}/prompts/${promptType}`, content, {
        headers: {
          'Content-Type': 'text/plain'
        }
      });
      
      if (promptType === 'inductive') {
        setInductivePrompt(prev => ({ 
          ...prev, 
          isCustom: response.data.is_custom 
        }));
      } else {
        setDeductivePrompt(prev => ({ 
          ...prev, 
          isCustom: response.data.is_custom 
        }));
      }
      
    } catch (err) {
      console.error(`Error saving ${promptType} prompt:`, err);
      setPromptError(`Failed to save ${promptType} prompt: ${err.response?.data?.detail || err.message}`);
    } finally {
      setPromptSaving(false);
    }
  };

  const resetPrompt = async (promptType) => {
    try {
      const response = await axios.delete(`${API_URL}/prompts/${promptType}`);
      // Fetch the default prompt after resetting
      fetchPrompt(promptType);
    } catch (err) {
      console.error(`Error resetting ${promptType} prompt:`, err);
      setPromptError(`Failed to reset ${promptType} prompt: ${err.response?.data?.detail || err.message}`);
    }
  };

  // Functions for codebase management
  const fetchCodebases = async () => {
    try {
      setCodebases(prev => ({ ...prev, loading: true }));
      setCodebaseError(null);
      
      const response = await axios.get(`${API_URL}/codebases/list`);
      setCodebases({
        default_codebases: response.data.default_codebases || [],
        user_codebases: response.data.user_codebases || [],
        loading: false
      });
      
      // Select the first codebase by default if none is selected
      if (!selectedCodebase && 
          (response.data.default_codebases?.length > 0 || response.data.user_codebases?.length > 0)) {
        const firstCodebase = response.data.default_codebases?.[0]?.filename || 
                              response.data.user_codebases?.[0]?.filename;
        if (firstCodebase) {
          setSelectedCodebase(firstCodebase);
          fetchCodebaseContent(firstCodebase);
        }
      }
    } catch (err) {
      console.error('Error fetching codebases:', err);
      setCodebaseError(`Failed to load codebases: ${err.response?.data?.detail || err.message}`);
      setCodebases(prev => ({ ...prev, loading: false }));
    }
  };

  const fetchCodebaseContent = async (codebaseName) => {
    if (!codebaseName) return;
    
    try {
      setCodebaseContents(prev => ({ ...prev, loading: true }));
      setCodebaseError(null);
      
      const response = await axios.get(`${API_URL}/codebases/${codebaseName}`);
      setCodebaseContents({
        codebase_name: response.data.codebase_name,
        is_default: response.data.is_default,
        codes: response.data.codes || [],
        loading: false
      });
    } catch (err) {
      console.error('Error fetching codebase content:', err);
      setCodebaseError(`Failed to load codebase content: ${err.response?.data?.detail || err.message}`);
      setCodebaseContents(prev => ({ ...prev, loading: false }));
    }
  };

  const createCodebase = async () => {
    if (!newCodebaseName) {
      setCodebaseError('Codebase name is required');
      return;
    }
    
    try {
      setSaving(true);
      setCodebaseError(null);
      
      const formData = new FormData();
      formData.append('codebase_name', newCodebaseName);
      if (baseCodebase) {
        formData.append('base_codebase', baseCodebase);
      }
      
      await axios.post(`${API_URL}/codebases/create`, formData);
      
      // Clear form and refresh the list
      setNewCodebaseName('');
      setBaseCodebase('');
      fetchCodebases();
      
    } catch (err) {
      console.error('Error creating codebase:', err);
      setCodebaseError(`Failed to create codebase: ${err.response?.data?.detail || err.message}`);
    } finally {
      setSaving(false);
    }
  };

  const addCodeToCodebase = async () => {
    if (!selectedCodebase) {
      setCodebaseError('Please select a codebase first');
      return;
    }
    
    if (!newCodeText) {
      setCodebaseError('Code text is required');
      return;
    }
    
    try {
      setSaving(true);
      setCodebaseError(null);
      
      const codeData = {
        text: newCodeText,
        metadata: {
          description: newCodeDescription
        }
      };
      
      await axios.post(`${API_URL}/codebases/${selectedCodebase}/add_code`, codeData);
      
      // Clear form and refresh
      setNewCodeText('');
      setNewCodeDescription('');
      fetchCodebaseContent(selectedCodebase);
      
    } catch (err) {
      console.error('Error adding code:', err);
      setCodebaseError(`Failed to add code: ${err.response?.data?.detail || err.message}`);
    } finally {
      setSaving(false);
    }
  };

  const deleteCodebase = async (codebaseName) => {
    try {
      await axios.delete(`${API_URL}/codebases/${codebaseName}`);
      
      // If we just deleted the selected codebase, clear selection
      if (selectedCodebase === codebaseName) {
        setSelectedCodebase(null);
        setCodebaseContents({ codes: [], is_default: false, loading: false });
      }
      
      // Refresh the list
      fetchCodebases();
      
    } catch (err) {
      console.error('Error deleting codebase:', err);
      setCodebaseError(`Failed to delete codebase: ${err.response?.data?.detail || err.message}`);
    }
  };

  // Load prompts when tab changes
  useEffect(() => {
    if (activeTab === 'inductive' && inductivePrompt.loading) {
      fetchPrompt('inductive');
    } else if (activeTab === 'deductive' && deductivePrompt.loading) {
      fetchPrompt('deductive');
    } else if (activeTab === 'codebases' && codebases.loading) {
      fetchCodebases();
    }
  }, [activeTab]);
  
  // Update selected codebase in config when it changes
  useEffect(() => {
    if (selectedCodebase) {
      setConfig(prev => ({
        ...prev,
        selected_codebase: selectedCodebase
      }));
    }
  }, [selectedCodebase]);
  
  // Fetch codebases when deductive mode is selected
  useEffect(() => {
    if (config.coding_mode === 'deductive' && codebases.loading) {
      fetchCodebases();
    }
  }, [config.coding_mode]);
  
  if (analyzing) {
    return <div className="loading">Analyzing file structure...</div>;
  }
  
  if (error) {
  return (
      <div className="error-panel">
        <h3>Error Analyzing File</h3>
        <p>{error}</p>
        <button onClick={onCancel}>Go Back</button>
      </div>
    );
  }

  return (
    <div className="file-config-form">
      <h2>Configure {file.filename}</h2>
      <p className="help-text">
        {file.is_default ? 'Adjust processing settings for this JSON file' : 'Configure how your JSON file should be processed'}
        <br/>
        <small className="json-note">JSON files contain structured data that can be mapped to specific fields</small>
      </p>
      
      <div className="config-tabs">
        <button 
          className={`tab-button ${activeTab === 'config' ? 'active' : ''}`}
          onClick={() => setActiveTab('config')}
        >
          File Configuration
        </button>
        <button 
          className={`tab-button ${activeTab === 'inductive' ? 'active' : ''}`}
          onClick={() => setActiveTab('inductive')}
        >
          Inductive Prompt
        </button>
        <button 
          className={`tab-button ${activeTab === 'deductive' ? 'active' : ''}`}
          onClick={() => setActiveTab('deductive')}
        >
          Deductive Prompt
        </button>
        <button 
          className={`tab-button ${activeTab === 'codebases' ? 'active' : ''}`}
          onClick={() => setActiveTab('codebases')}
        >
          Codebases
        </button>
      </div>
      
      {activeTab === 'config' && (
        <form onSubmit={handleSubmit}>
          <div className="form-section">
            <h3>Field Mappings</h3>
            
            <div className="form-group">
              <label>Content Field (main text):</label>
              <select 
                name="content_field" 
                value={config.content_field} 
                onChange={handleChange}
                required
              >
                <option value="">Select Content Field</option>
                {fileStructure?.fields?.map(field => (
                  <option 
                    key={field.name} 
                    value={field.name}
                    disabled={field.type !== "string"}
                  >
                    {field.name} {field.type !== "string" ? `(${field.type})` : ""}
                  </option>
                ))}
              </select>
              <small>This is the main text content that will be analyzed</small>
            </div>
            
            <div className="form-group">
              <label>List Field (if data is in an array):</label>
              <select 
                name="list_field" 
                value={config.list_field} 
                onChange={handleChange}
              >
                <option value="">None (data is not in an array)</option>
                <option value="root">Root (data is a top-level array)</option>
                {fileStructure?.arrays?.map(arrayField => (
                  <option key={arrayField} value={arrayField}>
                    {arrayField}
                  </option>
                ))}
              </select>
              <small>Select if your data is inside an array/list</small>
            </div>
            
            <div className="form-group">
              <label>Context Fields (additional data for context):</label>
              <div className="checkbox-group">
                {fileStructure?.fields?.map(field => (
                  field.name !== config.content_field && (
                    <label key={field.name} className="checkbox-label">
                      <input 
                        type="checkbox"
                        checked={config.context_fields.includes(field.name)}
                        onChange={(e) => handleContextFieldChange(field.name, e.target.checked)}
                      />
                      {field.name} ({field.type})
                    </label>
                  )
                ))}
              </div>
              <small>Additional fields to include as context for coding</small>
            </div>
          </div>
          
          <div className="form-section">
            <h3>Processing Settings</h3>
            
            <div className="form-group">
              <label>Coding Mode:</label>
              <select 
                name="coding_mode" 
                value={config.coding_mode} 
                onChange={handleChange}
              >
                <option value="inductive">Inductive (discover codes)</option>
                <option value="deductive">Deductive (use predefined codes)</option>
              </select>
            </div>
            
            {config.coding_mode === 'deductive' && (
              <div className="form-group">
                <label>Select Codebase:</label>
                <select
                  name="selected_codebase"
                  value={config.selected_codebase}
                  onChange={handleChange}
                  required={config.coding_mode === 'deductive'}
                >
                  <option value="">Select a codebase</option>
                  {codebases.default_codebases.map(codebase => (
                    <option key={`default-${codebase.filename}`} value={codebase.filename}>
                      {codebase.filename} ({codebase.code_count} codes) - Default
                    </option>
                  ))}
                  {codebases.user_codebases.map(codebase => (
                    <option key={`user-${codebase.filename}`} value={codebase.filename}>
                      {codebase.filename} ({codebase.code_count} codes)
                    </option>
                  ))}
                </select>
                {config.coding_mode === 'deductive' && !config.selected_codebase && (
                  <small className="validation-message">A codebase is required for deductive coding</small>
                )}
              </div>
            )}
            
            <div className="form-group checkbox-group">
              <label>
                <input 
                  type="checkbox" 
                  name="use_parsing" 
                  checked={config.use_parsing} 
                  onChange={handleChange}
                />
                Use Text Parsing
              </label>
              <small>Break down large texts into smaller meaning units</small>
            </div>
            
            <div className="form-group">
              <label>Segments Per Prompt:</label>
              <input 
                type="number" 
                name="preliminary_segments_per_prompt" 
                value={config.preliminary_segments_per_prompt} 
                onChange={handleChange}
                min="1"
                max="100"
              />
            </div>
            
            <div className="form-group">
              <label>Model:</label>
              <select 
                name="model_name" 
                value={config.model_name} 
                onChange={handleChange}
              >
                <option value="gpt-4o-mini">GPT-4o Mini</option>
                <option value="gpt-4o">GPT-4o</option>
                <option value="gpt-4">GPT-4</option>
                <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
              </select>
            </div>
          </div>
          
          <div className="form-actions">
            <button type="button" onClick={onCancel} className="secondary-button">
              Cancel
            </button>
            <button 
              type="submit" 
              className="start-button"
              disabled={config.coding_mode === 'deductive' && !config.selected_codebase}
            >
              Start Processing
            </button>
          </div>
        </form>
      )}
      
      {activeTab === 'inductive' && (
        <div className="prompt-editor">
          <div className="prompt-header">
            <h3>Inductive Coding Prompt</h3>
            {inductivePrompt.isCustom && (
              <span className="custom-badge">Custom</span>
            )}
          </div>
          
          {promptError && <div className="error">{promptError}</div>}
          
          {inductivePrompt.loading ? (
            <div className="loading">Loading prompt...</div>
          ) : (
            <>
              <textarea
                className="prompt-textarea"
                value={inductivePrompt.content}
                onChange={(e) => setInductivePrompt({ ...inductivePrompt, content: e.target.value })}
                rows={20}
              />
              <div className="prompt-actions">
                <button 
                  className="secondary-button"
                  onClick={() => resetPrompt('inductive')}
                  disabled={!inductivePrompt.isCustom || promptSaving}
                >
                  Reset to Default
                </button>
                <button 
                  className="primary-button"
                  onClick={() => savePrompt('inductive', inductivePrompt.content)}
                  disabled={promptSaving}
                >
                  {promptSaving ? 'Saving...' : 'Save Prompt'}
                </button>
              </div>
              <div className="prompt-info">
                <p>
                  This prompt template is used for inductive coding (discovering codes from data).
                  Customizing this prompt allows you to adjust how the AI model identifies and applies codes to your data.
        </p>
      </div>
            </>
          )}
        </div>
      )}
      
      {activeTab === 'deductive' && (
        <div className="prompt-editor">
          <div className="prompt-header">
            <h3>Deductive Coding Prompt</h3>
            {deductivePrompt.isCustom && (
              <span className="custom-badge">Custom</span>
            )}
          </div>
          
          {promptError && <div className="error">{promptError}</div>}
          
          {deductivePrompt.loading ? (
            <div className="loading">Loading prompt...</div>
          ) : (
            <>
              <textarea
                className="prompt-textarea"
                value={deductivePrompt.content}
                onChange={(e) => setDeductivePrompt({ ...deductivePrompt, content: e.target.value })}
                rows={20}
              />
              <div className="prompt-actions">
                <button 
                  className="secondary-button"
                  onClick={() => resetPrompt('deductive')}
                  disabled={!deductivePrompt.isCustom || promptSaving}
                >
                  Reset to Default
                </button>
                <button 
                  className="primary-button"
                  onClick={() => savePrompt('deductive', deductivePrompt.content)}
                  disabled={promptSaving}
                >
                  {promptSaving ? 'Saving...' : 'Save Prompt'}
                </button>
              </div>
              <div className="prompt-info">
                <p>
                  This prompt template is used for deductive coding (applying predefined codes).
                  Customizing this prompt allows you to adjust how the AI model applies your codebook to the data.
                </p>
              </div>
            </>
          )}
        </div>
      )}
      
      {activeTab === 'codebases' && (
        <div className="codebases-panel">
          {codebaseError && <div className="error">{codebaseError}</div>}
          
          <div className="codebases-layout">
            <div className="codebases-list-panel">
              <h3>Available Codebases</h3>
              
              {codebases.loading ? (
                <div className="loading">Loading codebases...</div>
              ) : (
                <div className="codebases-list">
                  <div className="codebases-section">
                    <h4>Default Codebases</h4>
                    <ul>
                      {codebases.default_codebases.map(codebase => (
                        <li 
                          key={codebase.filename}
                          className={selectedCodebase === codebase.filename ? 'selected' : ''}
                          onClick={() => {
                            setSelectedCodebase(codebase.filename);
                            fetchCodebaseContent(codebase.filename);
                          }}
                        >
                          <div className="codebase-item">
                            <div className="codebase-name">{codebase.filename}</div>
                            <div className="codebase-meta">
                              <span>{codebase.code_count} codes</span>
                              <span className="default-badge">Default</span>
                            </div>
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  <div className="codebases-section">
                    <h4>Your Codebases</h4>
                    <ul>
                      {codebases.user_codebases.length === 0 ? (
                        <li className="empty-list">No custom codebases yet</li>
                      ) : (
                        codebases.user_codebases.map(codebase => (
                          <li 
                            key={codebase.filename}
                            className={selectedCodebase === codebase.filename ? 'selected' : ''}
                            onClick={() => {
                              setSelectedCodebase(codebase.filename);
                              fetchCodebaseContent(codebase.filename);
                            }}
                          >
                            <div className="codebase-item">
                              <div className="codebase-name">{codebase.filename}</div>
                              <div className="codebase-meta">
                                <span>{codebase.code_count} codes</span>
                              </div>
                            </div>
                            <button 
                              className="delete-button"
                              onClick={(e) => {
                                e.stopPropagation();
                                deleteCodebase(codebase.filename);
                              }}
                              title="Delete codebase"
                            >
                              <TrashIcon />
                            </button>
                          </li>
                        ))
                      )}
                    </ul>
                  </div>
                  
                  <div className="create-codebase-section">
                    <h4>Create New Codebase</h4>
                    <div className="form-group">
                      <label>Codebase Name:</label>
                      <input 
                        type="text"
                        value={newCodebaseName}
                        onChange={(e) => setNewCodebaseName(e.target.value)}
                        placeholder="my_codebase.jsonl"
                      />
                    </div>
                    
                    <div className="form-group">
                      <label>Base Codebase (optional):</label>
                      <select
                        value={baseCodebase}
                        onChange={(e) => setBaseCodebase(e.target.value)}
                      >
                        <option value="">Create Empty Codebase</option>
                        {codebases.default_codebases.map(codebase => (
                          <option key={codebase.filename} value={codebase.filename}>
                            {codebase.filename} ({codebase.code_count} codes)
                          </option>
                        ))}
                        {codebases.user_codebases.map(codebase => (
                          <option key={codebase.filename} value={codebase.filename}>
                            {codebase.filename} ({codebase.code_count} codes)
                          </option>
                        ))}
                      </select>
                      <small>Start with an existing codebase as a template</small>
                    </div>
                    
                    <button 
                      className="secondary-button"
                      onClick={createCodebase}
                      disabled={saving || !newCodebaseName}
                    >
                      {saving ? 'Creating...' : 'Create Codebase'}
                    </button>
                  </div>
                </div>
              )}
            </div>
            
            <div className="codebase-content-panel">
              <h3>
                {selectedCodebase ? 
                  `Codes in ${selectedCodebase}` : 
                  'Select a codebase to view and edit codes'}
              </h3>
              
              {selectedCodebase && (
                <>
                  {codebaseContents.loading ? (
                    <div className="loading">Loading codes...</div>
                  ) : (
                    <>
                      <div className="codes-list">
                        {codebaseContents.codes.length === 0 ? (
                          <div className="empty-codes">
                            <p>No codes in this codebase</p>
                          </div>
                        ) : (
                          codebaseContents.codes.map((code, index) => (
                            <div key={index} className="code-item">
                              <div className="code-text">{code.text}</div>
                              <div className="code-description">{code.metadata?.description}</div>
                            </div>
                          ))
                        )}
                      </div>
                      
                      {/* Add new code form */}
                      <div className="add-code-form">
                        <h4>Add New Code</h4>
                        <div className="form-group">
                          <label>Code Name:</label>
                          <input
                            type="text"
                            value={newCodeText}
                            onChange={(e) => setNewCodeText(e.target.value)}
                            placeholder="Code name"
                          />
                        </div>
                        
                        <div className="form-group">
                          <label>Code Description:</label>
                          <textarea
                            value={newCodeDescription}
                            onChange={(e) => setNewCodeDescription(e.target.value)}
                            placeholder="Description of what this code represents"
                            rows={3}
                          />
                        </div>
                        
                        <button
                          className="primary-button"
                          onClick={addCodeToCodebase}
                          disabled={saving || !newCodeText}
                        >
                          {saving ? 'Adding...' : 'Add Code'}
                        </button>
                      </div>
                    </>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// JSON Document Viewer Component
function JsonDocumentViewer({ content }) {
  if (!content) return <div className="json-document-empty">Select a file to view its content</div>;
  
  return (
    <div className="json-document-viewer">
      <pre>{JSON.stringify(content, null, 2)}</pre>
    </div>
  );
}

function FileList({ files, onSelect, selectedFile, onDelete }) {
  const [viewingContent, setViewingContent] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSelect = async (file) => {
    onSelect(file);
    
    // Load file content for preview
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:8000/files/${file.filename}/content`);
      if (!response.ok) {
        throw new Error(`Failed to load file content: ${response.statusText}`);
      }
      const data = await response.json();
      setViewingContent(data.content);
    } catch (error) {
      console.error("Error loading file content:", error);
      setViewingContent(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="file-list-container">
      <div className="files-section">
        <h3>Available Files</h3>
        <div className="file-groups">
          <div className="file-group">
            <h4>Uploaded Files</h4>
            <ul className="file-list">
              {files.user_files?.length > 0 ? (
                files.user_files.map((file) => (
                  <li
                    key={file.filename}
                    className={selectedFile?.filename === file.filename ? 'selected' : ''}
                    onClick={() => handleSelect(file)}
                  >
                    <span className="file-icon">
                      <FontAwesomeIcon icon={faFileCode} />
                    </span>
                    <span className="file-name">{file.filename}</span>
                    <span className="file-actions">
                      <button 
                        className="action-button delete-button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDelete(file.filename);
                        }}
                      >
                        <FontAwesomeIcon icon={faTrash} />
                      </button>
                    </span>
                  </li>
                ))
              ) : (
                <li className="empty-message">No uploaded files</li>
              )}
            </ul>
          </div>
          <div className="file-group">
            <h4>Default Files</h4>
            <ul className="file-list">
              {files.default_files?.length > 0 ? (
                files.default_files.map((file) => (
                  <li
                    key={file.filename}
                    className={selectedFile?.filename === file.filename ? 'selected' : ''}
                    onClick={() => handleSelect(file)}
                  >
                    <span className="file-icon">
                      <FontAwesomeIcon icon={faFileCode} />
                    </span>
                    <span className="file-name">{file.filename}</span>
                  </li>
                ))
              ) : (
                <li className="empty-message">No default files</li>
              )}
            </ul>
          </div>
        </div>
      </div>
      
      <div className="file-content-preview">
        <h3>File Preview</h3>
        {loading ? (
          <div className="loading-content">Loading content...</div>
        ) : (
          <JsonDocumentViewer content={viewingContent} />
        )}
      </div>
    </div>
  );
}

// FileUpload Component
function FileUpload({ onUploadSuccess, setUploadProgress, setUploadStatus }) {
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = () => {
    setDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  };

  const handleClickUpload = () => {
    fileInputRef.current.click();
  };

  const handleFileUpload = async (file) => {
    // Validate file type
    if (!file.name.endsWith('.json')) {
      setUploadStatus('Error: Only JSON files are supported');
      setTimeout(() => setUploadStatus(''), 3000);
      return;
    }

    setUploadStatus('Uploading file...');
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post(`${API_URL}/files/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percentCompleted);
        }
      });

      setUploadStatus('File uploaded successfully!');
      setUploadProgress(100);
      
      // Reset status after a delay
      setTimeout(() => {
        setUploadStatus('');
        setUploadProgress(0);
      }, 3000);

      // Refresh file list
      if (onUploadSuccess) {
        onUploadSuccess();
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadStatus(`Error: ${error.response?.data?.detail || 'Failed to upload file'}`);
      setUploadProgress(0);
    }
  };

  return (
    <div 
      className={`file-upload-area ${dragging ? 'dragging' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClickUpload}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".json"
        style={{ display: 'none' }}
      />
      <FontAwesomeIcon icon={faUpload} className="upload-icon" />
      <div className="upload-text">
        <p>Drag & drop a JSON file here or click to browse</p>
        <p className="file-type-note">Only .json files are supported</p>
      </div>
    </div>
  );
}

// Helper function to format dates
const formatDate = (dateString) => {
  if (!dateString) return 'N/A';
  
  const date = new Date(dateString);
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

// Helper function to download output
const downloadOutput = (jobId) => {
  window.open(`${API_URL}/jobs/${jobId}/output`, '_blank');
};

// Helper function to download validation
const downloadValidation = (jobId) => {
  window.open(`${API_URL}/jobs/${jobId}/validation`, '_blank');
};

function App() {
  // State for file selection
  const [files, setFiles] = useState({ user_files: [], default_files: [] });
  const [selectedFile, setSelectedFile] = useState(null);
  const [showFileConfig, setShowFileConfig] = useState(false);
  const [analyzeLoading, setAnalyzeLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('');
  const [jobs, setJobs] = useState([]);
  const [activeJobId, setActiveJobId] = useState(null);
  
  // File drag and drop state
  const [dragging, setDragging] = useState(false);

  // State for codebase management
  const [codebases, setCodebases] = useState({ default_codebases: [], user_codebases: [] });
  const [selectedCodebase, setSelectedCodebase] = useState(null);
  const [newCodeText, setNewCodeText] = useState('');
  const [newCodeDescription, setNewCodeDescription] = useState('');
  
  // Polling for job status
  useEffect(() => {
    const intervalId = setInterval(() => {
      fetchJobStatus();
    }, 5000);
    
    return () => clearInterval(intervalId);
  }, []);
  
  // Fetch available files on component mount
  useEffect(() => {
    fetchFiles();
    fetchCodebases();
  }, []);

  // Function to fetch available files
  const fetchFiles = async () => {
    try {
      const response = await axios.get(`${API_URL}/files/list`);
      setFiles(response.data);
    } catch (error) {
      console.error('Error fetching files:', error);
    }
  };

  // Function to fetch available codebases
  const fetchCodebases = async () => {
    try {
      const response = await axios.get(`${API_URL}/codebases/list`);
      setCodebases(response.data);
    } catch (error) {
      console.error('Error fetching codebases:', error);
    }
  };

  // Function to fetch job status
  const fetchJobStatus = async () => {
    try {
      const response = await axios.get(`${API_URL}/jobs`);
      setJobs(response.data.jobs || []);
    } catch (error) {
      console.error('Error fetching job status:', error);
    }
  };

  // File select handler
  const handleFileSelect = (file) => {
    setSelectedFile(file);
  };

  // File delete handler
  const handleDeleteFile = async (filename) => {
    if (window.confirm(`Are you sure you want to delete ${filename}?`)) {
      try {
        await axios.delete(`${API_URL}/files/${filename}`);
        fetchFiles(); // Refresh the file list
        if (selectedFile?.filename === filename) {
          setSelectedFile(null);
        }
      } catch (error) {
        console.error('Error deleting file:', error);
        alert(`Failed to delete file: ${error.message}`);
      }
    }
  };

  // Analyze file structure 
  const handleAnalyzeFile = async (filename) => {
    if (!filename) return;
    
    setAnalyzeLoading(true);
    try {
      const response = await axios.get(`${API_URL}/analyze-file/${filename}`);
      
      // If successful, proceed to configuration
      setShowFileConfig(true);
    } catch (error) {
      console.error('Error analyzing file:', error);
      alert(`Failed to analyze file: ${error.response?.data?.detail || error.message}`);
    } finally {
      setAnalyzeLoading(false);
    }
  };

  // Handle configuration submission
  const handleConfigSubmit = async (config) => {
    try {
      const response = await axios.post(`${API_URL}/run-pipeline-with-config`, config);
      setShowFileConfig(false);
      // Refresh jobs list
      fetchJobStatus();
      setActiveJobId(response.data.job_id);
    } catch (error) {
      console.error('Error submitting configuration:', error);
      alert(`Failed to start processing: ${error.response?.data?.detail || error.message}`);
    }
  };

  // Cancel configuration
  const handleCancelConfig = () => {
    setShowFileConfig(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>LLM Qualitative Coder</h1>
        <p>Automated qualitative coding using Large Language Models</p>
      </header>
      
      <div className="content">
        {showFileConfig ? (
          // Show file configuration form when a file is being configured
          <FileConfigForm 
            file={selectedFile}
            onSubmit={handleConfigSubmit}
            onCancel={handleCancelConfig}
          />
        ) : (
          <div className="main-panel">
            <section className="files-section">
              <h2>Input Files</h2>
              <FileUpload 
                onUploadSuccess={fetchFiles}
                setUploadProgress={setUploadProgress}
                setUploadStatus={setUploadStatus}
              />
              
              <div className="upload-status">
                {uploadProgress > 0 && uploadProgress < 100 && (
                  <div className="progress-bar">
                    <div className="progress" style={{width: `${uploadProgress}%`}}></div>
                  </div>
                )}
                {uploadStatus && <p className="status-message">{uploadStatus}</p>}
              </div>
              
              <FileList 
                files={files}
                onSelect={handleFileSelect}
                selectedFile={selectedFile}
                onDelete={handleDeleteFile}
              />
              
              {selectedFile && (
                <div className="file-actions">
                  <button className="primary-button" onClick={() => handleAnalyzeFile(selectedFile.filename)}>
                    {analyzeLoading ? 'Analyzing...' : 'Analyze Structure'}
                  </button>
                  <button className="secondary-button" onClick={() => setShowFileConfig(true)}>
                    Configure Processing
                  </button>
                </div>
              )}
            </section>

            <section className="jobs-section">
              <h2>Processing Jobs</h2>
              
              <div className="jobs-list">
                {jobs.length === 0 ? (
                  <p>No jobs yet. Start a new pipeline.</p>
                ) : (
                  <ul>
                    {jobs.map(job => (
                      <li 
                        key={job.job_id}
                        className={activeJobId === job.job_id ? 'selected' : ''}
                        onClick={() => setActiveJobId(job.job_id)}
                      >
                        <div className={`job-status ${job.status}`}></div>
                        <div className="job-info">
                          <div className="job-name">{job.filename || 'Job'}</div>
                          <div className="job-meta">
                            <span className="job-date">{formatDate(job.started_at)}</span>
                            <span className={`job-status-text ${job.status}`}>{job.status}</span>
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              
              {activeJobId && jobs.find(j => j.job_id === activeJobId) && (
                <div className="job-details">
                  <h3>Job Details</h3>
                  <div><strong>Status:</strong> {jobs.find(j => j.job_id === activeJobId).status}</div>
                  <div><strong>Started:</strong> {formatDate(jobs.find(j => j.job_id === activeJobId).started_at)}</div>
                  
                  {jobs.find(j => j.job_id === activeJobId).completed_at && (
                    <div><strong>Completed:</strong> {formatDate(jobs.find(j => j.job_id === activeJobId).completed_at)}</div>
                  )}
                  
                  {jobs.find(j => j.job_id === activeJobId).error && (
                    <div className="error">Error: {jobs.find(j => j.job_id === activeJobId).error}</div>
                  )}
                  
                  {jobs.find(j => j.job_id === activeJobId).status === 'completed' && (
                    <div className="job-actions">
                      <button onClick={() => downloadOutput(activeJobId)}>
                        Download Output
                      </button>
                      <button onClick={() => downloadValidation(activeJobId)}>
                        Download Validation
                      </button>
                    </div>
                  )}
                </div>
              )}
            </section>
          </div>
        )}
      </div>
    </div>
  );
}

export default App
