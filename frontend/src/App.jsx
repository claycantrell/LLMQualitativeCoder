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
  const [activeTab, setActiveTab] = useState('config');
  const [fileStructure, setFileStructure] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [fileContent, setFileContent] = useState(null);
  const [loadingContent, setLoadingContent] = useState(false);
  
  // Add state for preview tabs
  const [previewTab, setPreviewTab] = useState('file');
  const [promptPreview, setPromptPreview] = useState('');
  
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
  const [inductivePrompt, setInductivePrompt] = useState({
    loading: true,
    content: '',
    isCustom: false,
    isDirty: false
  });
  
  const [deductivePrompt, setDeductivePrompt] = useState({
    loading: true, 
    content: '',
    isCustom: false,
    isDirty: false
  });
  
  // Codebase management state
  const [codebases, setCodebases] = useState({ default_codebases: [], user_codebases: [] });
  const [selectedCodebase, setSelectedCodebase] = useState(null);
  const [newCodeText, setNewCodeText] = useState('');
  const [newCodeDescription, setNewCodeDescription] = useState('');
  const [newCodebaseName, setNewCodebaseName] = useState('');
  const [selectedBaseCodebase, setSelectedBaseCodebase] = useState('');
  
  // Load file structure on component mount
  useEffect(() => {
    // Fetch file structure data
    const fetchFileStructure = async () => {
      try {
        setAnalyzing(true);
              const response = await axios.get(`http://localhost:8000/analyze-file/${file.filename}`);
      setFileStructure(response.data.structure);
      
      // Log the structure to help with debugging
      console.log("File structure:", response.data);
        
              // Pre-fill config with suggested mappings
      if (response.data.suggested_mappings) {
        setConfig(prev => ({
          ...prev,
          content_field: response.data.suggested_mappings.content_field || '',
          context_fields: response.data.suggested_mappings.context_fields || [],
          list_field: response.data.suggested_mappings.list_field || ''
        }));
        }
        
        setAnalyzing(false);
      } catch (err) {
        console.error('Error fetching file structure:', err);
        setError('Failed to analyze file structure. Please try again.');
        setAnalyzing(false);
      }
    };
    
    // Fetch the file content for preview
    const fetchFileContent = async () => {
      try {
        setLoadingContent(true);
        const response = await fetch(`http://localhost:8000/files/${file.filename}/content`);
        if (!response.ok) {
          throw new Error(`Failed to load file content: ${response.statusText}`);
        }
        const data = await response.json();
        setFileContent(data);
        setLoadingContent(false);
      } catch (error) {
        console.error('Error loading file content:', error);
        setLoadingContent(false);
      }
    };

    fetchFileStructure();
    fetchFileContent();
    fetchCodebases();
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
    if (e) e.preventDefault();
    
    // Create dynamic config that should work for all file types
    const dynamicConfig = {
      file_id: file.filename,
      content_field: config.content_field,
      context_fields: config.context_fields,
      list_field: config.list_field,
      coding_mode: config.coding_mode,
      use_parsing: config.use_parsing,
      preliminary_segments_per_prompt: config.preliminary_segments_per_prompt,
      meaning_units_per_assignment_prompt: config.meaning_units_per_assignment_prompt,
      context_size: config.context_size,
      model_name: config.model_name,
      temperature: config.temperature,
      max_tokens: config.max_tokens,
      thread_count: config.thread_count,
      selected_codebase: config.selected_codebase
    };
    
    console.log("Submitting configuration:", dynamicConfig);
    onSubmit(dynamicConfig, false);
  };
  
  // Add functions to fetch and save prompts
  const fetchPrompt = async (promptType) => {
    try {
      const response = await axios.get(`${API_URL}/prompts/${promptType}`);
      if (promptType === 'inductive') {
        setInductivePrompt({
          content: response.data.content,
          isCustom: response.data.is_custom,
          loading: false,
          isDirty: false
        });
      } else {
        setDeductivePrompt({
          content: response.data.content,
          isCustom: response.data.is_custom,
          loading: false,
          isDirty: false
        });
      }
    } catch (err) {
      console.error(`Error fetching ${promptType} prompt:`, err);
      // Set default content in case of error
      const defaultContent = promptType === 'inductive' 
        ? "Error loading inductive prompt. Using default prompt."
        : "Error loading deductive prompt. Using default prompt.";
        
      if (promptType === 'inductive') {
        setInductivePrompt({
          content: defaultContent,
          isCustom: false,
          loading: false,
          isDirty: false
        });
      } else {
        setDeductivePrompt({
          content: defaultContent,
          isCustom: false,
          loading: false,
          isDirty: false
        });
      }
    }
  };

  const savePrompt = async (promptType, content) => {
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
    }
  };

  const resetPrompt = async (promptType) => {
    try {
      const response = await axios.delete(`${API_URL}/prompts/${promptType}`);
      // Fetch the default prompt after resetting
      fetchPrompt(promptType);
    } catch (err) {
      console.error(`Error resetting ${promptType} prompt:`, err);
    }
  };

  // Functions for codebase management
  const fetchCodebases = async () => {
    try {
      const response = await axios.get(`${API_URL}/codebases/list`);
      setCodebases(response.data);
    } catch (error) {
      console.error('Error fetching codebases:', error);
    }
  };

  const createCodebase = async () => {
    if (!newCodebaseName) {
      setError('Codebase name is required');
      return;
    }
    
    try {
      // Create FormData object to match the backend's Form parameters expectation
      const formData = new FormData();
      formData.append('codebase_name', newCodebaseName);
      if (selectedBaseCodebase) {
        formData.append('base_codebase', selectedBaseCodebase);
      }
      
      const response = await axios.post(`${API_URL}/codebases/create`, formData);
      
      // Clear form and refresh the list
      setNewCodebaseName('');
      setSelectedBaseCodebase('');
      fetchCodebases();
      
    } catch (err) {
      console.error('Error creating codebase:', err);
      setError(`Failed to create codebase: ${err.response?.data?.detail || err.message}`);
    }
  };

  const addCodeToCodebase = async () => {
    if (!selectedCodebase) {
      setError('Please select a codebase first');
      return;
    }
    
    if (!newCodeText) {
      setError('Code text is required');
      return;
    }
    
    try {
      const response = await axios.post(`${API_URL}/codebases/${selectedCodebase}/add_code`, {
        text: newCodeText,
        metadata: {
          description: newCodeDescription
        }
      });
      
      // Clear form and refresh
      setNewCodeText('');
      setNewCodeDescription('');
      
      // Refresh the codebases list
      fetchCodebases();
      
    } catch (err) {
      console.error('Error adding code:', err);
      setError(`Failed to add code: ${err.response?.data?.detail || err.message}`);
    }
  };

  const deleteCodebase = async (codebaseName) => {
    try {
      await axios.delete(`${API_URL}/codebases/${codebaseName}`);
      
      // If we just deleted the selected codebase, clear selection
      if (selectedCodebase === codebaseName) {
        setSelectedCodebase(null);
      }
      
      // Refresh the list
      fetchCodebases();
      
    } catch (err) {
      console.error('Error deleting codebase:', err);
      setError(`Failed to delete codebase: ${err.response?.data?.detail || err.message}`);
    }
  };

  // Load prompts when tab changes
  useEffect(() => {
    if (activeTab === 'inductive' && inductivePrompt.loading) {
      fetchPrompt('inductive');
      
      // Add timeout to ensure loading state is cleared
      const timeout = setTimeout(() => {
        setInductivePrompt(prev => {
          if (prev.loading) {
            return {
              ...prev,
              loading: false,
              content: "Timed out loading the prompt. Please try again."
            };
          }
          return prev;
        });
      }, 5000);
      
      return () => clearTimeout(timeout);
    } else if (activeTab === 'deductive' && deductivePrompt.loading) {
      fetchPrompt('deductive');
      
      // Add timeout to ensure loading state is cleared
      const timeout = setTimeout(() => {
        setDeductivePrompt(prev => {
          if (prev.loading) {
            return {
              ...prev,
              loading: false,
              content: "Timed out loading the prompt. Please try again."
            };
          }
          return prev;
        });
      }, 5000);
      
      return () => clearTimeout(timeout);
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
  
  // Form validation
  const isFormValid = () => {
    // For deductive coding, a codebase is required
    if (config.coding_mode === 'deductive' && !config.selected_codebase) {
      console.log("Form invalid: deductive mode requires a codebase");
      return false;
    }
    
    // For most files, a content field is required
    // But for some known formats, we might not need it
    if (!config.content_field && !file.is_default) {
      console.log("Form invalid: content field required for custom files");
      return false;
    }
    
    console.log("Form validation passed");
    return true;
  };
  
  // Add a function to generate the prompt preview based on current settings
  useEffect(() => {
    if (fileContent && config.content_field) {
      generatePromptPreview();
    }
  }, [
    config.content_field, 
    config.context_fields, 
    config.list_field, 
    config.coding_mode, 
    config.selected_codebase,
    config.meaning_units_per_assignment_prompt,
    config.preliminary_segments_per_prompt,
    config.use_parsing,
    config.context_size,
    fileContent,
    file.filename
  ]);
  
  const generatePromptPreview = () => {
    // Only generate preview if we have the necessary data
    if (!fileContent || !config.content_field) {
      setPromptPreview("Configure the required fields first to see prompt preview");
      return;
    }
    
    // Set loading state
    setPromptPreview("Loading prompt preview...");
    
    // Create a config object to send to the API - include all relevant settings
    const previewConfig = {
      file_id: file.filename,
      coding_mode: config.coding_mode,
      content_field: config.content_field,
      context_fields: config.context_fields,
      list_field: config.list_field,
      selected_codebase: config.selected_codebase,
      meaning_units_per_assignment_prompt: config.meaning_units_per_assignment_prompt,
      preliminary_segments_per_prompt: config.preliminary_segments_per_prompt,
      use_parsing: config.use_parsing,
      context_size: config.context_size,
      model_name: config.model_name,
      temperature: config.temperature,
      max_tokens: config.max_tokens,
      thread_count: config.thread_count
    };
    
    // Call the API endpoint
    axios.post(`${API_URL}/preview-prompt`, previewConfig)
      .then(response => {
        setPromptPreview(response.data.prompt);
      })
      .catch(error => {
        console.error("Error fetching prompt preview:", error);
        setPromptPreview("Error fetching prompt preview. Please ensure all required fields are configured correctly.");
      });
  };
  
  if (analyzing) {
    return <div className="loading">Analyzing file structure...</div>;
  }
  
  if (error) {
  return (
      <div className="error-panel">
        <h3>Error</h3>
        <p>{error}</p>
        <button onClick={() => onCancel()}>Go Back</button>
      </div>
    );
  }

  return (
    <div className="file-config-form">
      <h2>Configure Processing for {file.filename}</h2>
      <p className="help-text">Set up how your file should be processed by the qualitative coding pipeline.</p>
      
      <div className="config-tabs">
        <button 
          className={`tab-button ${activeTab === 'config' ? 'active' : ''}`}
          onClick={() => setActiveTab('config')}
        >
          Configuration
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
        <>
          {analyzing ? (
            <div className="loading">Analyzing file structure...</div>
          ) : error ? (
            <div className="error-panel">
              <h3>Error</h3>
              <p>{error}</p>
              <button onClick={() => onCancel()}>Go Back</button>
      </div>
          ) : (
            <>
              <div className="file-config-layout">
                <div className="file-mapping-section">
                  <div className="form-section">
                    <h3>Field Mapping</h3>
                    <div className="form-group">
                      <label htmlFor="content-field">Content Field:</label>
                      <select 
                        id="content-field" 
                        value={config.content_field}
                        onChange={e => setConfig({...config, content_field: e.target.value})}
                        required
                      >
                        <option value="">-- Select Field --</option>
                        {fileStructure?.fields?.map(field => (
                          <option key={field.name} value={field.name}>{field.name}</option>
                        ))}
                      </select>
                      <small>The field containing the main text content to analyze</small>
                    </div>
                    
                    <div className="form-group">
                      <label>Context Fields:</label>
                      <div className="checkbox-group">
                        {fileStructure?.fields?.map(field => (
                          <label key={field.name} className="checkbox-label">
                            <input 
                              type="checkbox"
                              checked={config.context_fields.includes(field.name)}
                              onChange={e => {
                                if (e.target.checked) {
                                  setConfig({...config, context_fields: [...config.context_fields, field.name]});
                                } else {
                                  setConfig({...config, context_fields: config.context_fields.filter(f => f !== field.name)});
                                }
                              }}
                            />
                            {field.name}
                          </label>
                        ))}
                      </div>
                      <small>Additional fields to provide as context</small>
                    </div>
                    
                    {fileStructure?.arrays?.length > 0 && (
                      <div className="form-group">
                        <label htmlFor="list-field">Nested Data Field:</label>
                        <select 
                          id="list-field" 
                          value={config.list_field}
                          onChange={e => setConfig({...config, list_field: e.target.value})}
                        >
                          <option value="">-- None --</option>
                          {fileStructure?.fields?.map(field => (
                            <option key={field.name} value={field.name}>{field.name}</option>
                          ))}
                        </select>
                        <small>If your data has nested arrays, select the field containing the nested data</small>
                      </div>
                    )}
                  </div>
                  
                  <div className="form-section">
                    <h3>Coding Options</h3>
                    <div className="form-group">
                      <label htmlFor="coding-mode">Coding Mode:</label>
                      <select 
                        id="coding-mode" 
                        value={config.coding_mode}
                        onChange={e => setConfig({...config, coding_mode: e.target.value})}
                      >
                        <option value="inductive">Inductive (generate codes from data)</option>
                        <option value="deductive">Deductive (use predefined codes)</option>
                      </select>
                    </div>
                    
                    {config.coding_mode === 'deductive' && (
                      <div className="form-group">
                        <label htmlFor="selected-codebase">Select Codebase:</label>
                        <select 
                          id="selected-codebase" 
                          value={config.selected_codebase}
                          onChange={e => setConfig({...config, selected_codebase: e.target.value})}
                          required
                        >
                          <option value="">-- Select Codebase --</option>
                          {codebases.default_codebases?.map(codebase => (
                            <option key={codebase.filename} value={codebase.filename}>
                              {codebase.filename} (Default)
                            </option>
                          ))}
                          {codebases.user_codebases?.map(codebase => (
                            <option key={codebase.filename} value={codebase.filename}>
                              {codebase.filename}
                            </option>
                          ))}
                        </select>
                        {config.coding_mode === 'deductive' && !config.selected_codebase && (
                          <span className="validation-message">You must select a codebase for deductive coding</span>
                        )}
                      </div>
                    )}
                    
                    <div className="form-group">
                      <label className="checkbox-label">
                        <input 
                          type="checkbox"
                          checked={config.use_parsing}
                          onChange={e => setConfig({...config, use_parsing: e.target.checked})}
                        />
                        Use LLM for preliminary segmentation
                      </label>
                      <small>Split text into meaningful segments before coding</small>
                    </div>
                  </div>
                  
                  <div className="form-section">
                    <h3>LLM Settings</h3>
                    <div className="form-row">
                      <div className="form-group">
                        <label htmlFor="model">Model:</label>
                        <select 
                          id="model" 
                          value={config.model_name}
                          onChange={e => setConfig({...config, model_name: e.target.value})}
                        >
                          <option value="gpt-4o-mini">GPT-4o-mini (Faster)</option>
                          <option value="gpt-4o">GPT-4o (Better quality)</option>
                        </select>
                      </div>
                      
                      <div className="form-group">
                        <label htmlFor="temperature">Temperature:</label>
                        <input 
                          type="number" 
                          id="temperature" 
                          min="0" 
                          max="2" 
                          step="0.1"
                          value={config.temperature}
                          onChange={e => setConfig({...config, temperature: parseFloat(e.target.value)})}
                        />
                      </div>
                      
                      <div className="form-group">
                        <label htmlFor="max-tokens">Max Tokens:</label>
                        <input 
                          type="number" 
                          id="max-tokens" 
                          min="100" 
                          max="4000" 
                          step="100"
                          value={config.max_tokens}
                          onChange={e => setConfig({...config, max_tokens: parseInt(e.target.value)})}
                        />
                      </div>
                    </div>
                    
                    <div className="form-row">
                      <div className="form-group">
                        <label htmlFor="thread-count">Thread Count:</label>
                        <input 
                          type="number" 
                          id="thread-count" 
                          min="1" 
                          max="10" 
                          value={config.thread_count}
                          onChange={e => setConfig({...config, thread_count: parseInt(e.target.value)})}
                        />
                        <small>Number of parallel requests</small>
                      </div>
                      
                      <div className="form-group">
                        <label htmlFor="segments-per-prompt">Segments Per Parsing Prompt:</label>
                        <input 
                          type="number" 
                          id="segments-per-prompt" 
                          min="1" 
                          max="10" 
                          value={config.preliminary_segments_per_prompt}
                          onChange={e => setConfig({...config, preliminary_segments_per_prompt: parseInt(e.target.value)})}
                        />
                      </div>
                      
                      <div className="form-group">
                        <label htmlFor="units-per-prompt">Meaning Units Per Assignment:</label>
                        <input 
                          type="number" 
                          id="units-per-prompt" 
                          min="1" 
                          max="20" 
                          value={config.meaning_units_per_assignment_prompt}
                          onChange={e => setConfig({...config, meaning_units_per_assignment_prompt: parseInt(e.target.value)})}
                        />
                        <small>Number of meaning units to include in each prompt</small>
                      </div>
                      
                      <div className="form-group">
                        <label htmlFor="context-size">Context Window Size:</label>
                        <input 
                          type="number" 
                          id="context-size" 
                          min="1" 
                          max="20" 
                          value={config.context_size}
                          onChange={e => setConfig({...config, context_size: parseInt(e.target.value)})}
                        />
                        <small>Number of surrounding meaning units to include as context</small>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="file-preview-section">
                  <div className="preview-tabs">
                    <button 
                      className={`tab-button ${previewTab === 'file' ? 'active' : ''}`}
                      onClick={() => setPreviewTab('file')}
                    >
                      File Preview
        </button>
                    <button 
                      className={`tab-button ${previewTab === 'prompt' ? 'active' : ''}`}
                      onClick={() => setPreviewTab('prompt')}
                    >
                      Prompt Preview
                    </button>
                  </div>
                  
                  <div className="json-document-container">
                    {loadingContent ? (
                      <div className="loading-content">Loading content...</div>
                    ) : (
                      <>
                        {previewTab === 'file' ? (
                          <JsonDocumentViewer content={fileContent} />
                        ) : (
                          <div className="prompt-preview">
                            <pre>{promptPreview}</pre>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </div>
              
              <div className="form-actions">
                <button 
                  className="secondary-button" 
                  onClick={() => onCancel()}
                >
                  Cancel
                </button>
                <button 
                  className="primary-button" 
                  onClick={handleSubmit}
                  disabled={!isFormValid()}
                >
                  Start Processing
                </button>
              </div>
            </>
          )}
        </>
      )}
      
      {activeTab === 'inductive' && (
        <PromptEditor 
          type="inductive"
          prompt={inductivePrompt}
          setPrompt={setInductivePrompt}
          fetchPrompt={fetchPrompt}
          savePrompt={savePrompt}
          resetPrompt={resetPrompt}
        />
      )}
      
      {activeTab === 'deductive' && (
        <PromptEditor 
          type="deductive"
          prompt={deductivePrompt}
          setPrompt={setDeductivePrompt}
          fetchPrompt={fetchPrompt}
          savePrompt={savePrompt}
          resetPrompt={resetPrompt}
        />
      )}
      
      {activeTab === 'codebases' && (
        <CodebaseManager 
          codebases={codebases}
          selectedCodebase={selectedCodebase}
          setSelectedCodebase={setSelectedCodebase}
          newCodeText={newCodeText}
          setNewCodeText={setNewCodeText}
          newCodeDescription={newCodeDescription}
          setNewCodeDescription={setNewCodeDescription}
          newCodebaseName={newCodebaseName}
          setNewCodebaseName={setNewCodebaseName}
          selectedBaseCodebase={selectedBaseCodebase}
          setSelectedBaseCodebase={setSelectedBaseCodebase}
          handleCreateCodebase={createCodebase}
          handleAddCode={addCodeToCodebase}
          handleSelectCodebase={deleteCodebase}
          fetchCodebases={fetchCodebases}
        />
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
      
      <div className="file-preview-container">
        <h3>File Preview</h3>
        <div className="file-content-preview">
          {loading ? (
            <div className="loading-content">Loading content...</div>
          ) : (
            <JsonDocumentViewer content={viewingContent} />
          )}
        </div>
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

// Add PromptEditor component definition
function PromptEditor({ type, prompt, setPrompt, fetchPrompt, savePrompt, resetPrompt }) {
  const handlePromptChange = (e) => {
    setPrompt({
      ...prompt,
      content: e.target.value,
      isDirty: true
    });
  };

  const handleSave = async () => {
    await savePrompt(type, prompt.content);
    setPrompt({
      ...prompt,
      isDirty: false
    });
  };

  const handleReset = async () => {
    if (window.confirm(`Are you sure you want to reset the ${type} prompt to default?`)) {
      await resetPrompt(type);
    }
  };

  return (
    <div className="prompt-editor">
      <h3>{type.charAt(0).toUpperCase() + type.slice(1)} Prompt {prompt.isCustom && <span className="custom-indicator">(Custom)</span>}</h3>
      
      <div className="prompt-info">
        <p>
          {type === 'inductive' 
            ? 'The inductive prompt is used when generating codes from data.' 
            : 'The deductive prompt is used when applying predefined codes to data.'}
        </p>
      </div>
      
      <textarea 
        className="prompt-textarea"
        value={prompt.content || ''}
        onChange={handlePromptChange}
        rows={15}
      />
      
      <div className="prompt-actions">
        <button 
          className="secondary-button" 
          onClick={handleReset}
        >
          Reset to Default
        </button>
        <button 
          className="primary-button" 
          onClick={handleSave}
          disabled={!prompt.isDirty}
        >
          Save Changes
        </button>
      </div>
    </div>
  );
}

// Add CodebaseManager component definition
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
  handleSelectCodebase,
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
  }, [activeCodebase, codebases]);
  
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
  const handleCodebaseSelect = (codebaseName) => {
    setActiveCodebase(codebaseName);
    setSelectedCodebase(codebaseName);
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
                    onClick={() => handleCodebaseSelect(codebase.filename)}
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
                    onClick={() => handleCodebaseSelect(codebase.filename)}
                  >
                    {codebase.filename} <span className="code-count">({codebase.code_count} codes)</span>
                    <button 
                      className="delete-button"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (window.confirm(`Are you sure you want to delete ${codebase.filename}?`)) {
                          handleSelectCodebase(codebase.filename);
                          fetchCodebases();
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
  const handleConfigSubmit = async (config, isStandardConfig) => {
    try {
      console.log("Sending configuration to backend:", config);
      const endpoint = isStandardConfig ? `${API_URL}/run-pipeline` : `${API_URL}/run-pipeline-with-config`;
      
      const response = await axios.post(endpoint, config);
      console.log("Processing job started:", response.data);
      
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

  // When showing the file config form, also analyze the file structure
  const showConfigAndAnalyze = (file) => {
    setShowFileConfig(true);
    handleAnalyzeFile(file.filename);
  };

  return (
    <div className="App">
      <div className="content-wrapper">
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
                  <button className="primary-button" onClick={() => showConfigAndAnalyze(selectedFile)}>
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
    </div>
  );
}

export default App
