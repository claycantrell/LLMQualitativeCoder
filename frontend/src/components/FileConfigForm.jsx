import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_URL } from '../config';
import PromptEditor from './PromptEditor';
import CodebaseManager from './CodebaseManager';
import JsonDocumentViewer from './JsonDocumentViewer';
import ExpandableContainer from './ExpandableContainer';

function FileConfigForm({ file, onSubmit, onCancel }) {
  const [activeTab, setActiveTab] = useState('config');
  const [fileStructure, setFileStructure] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [fileContent, setFileContent] = useState(null);
  const [loadingContent, setLoadingContent] = useState(false);
  
  const [previewTab, setPreviewTab] = useState('file');
  const [promptPreview, setPromptPreview] = useState('');
  
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
  
  const [parsePrompt, setParsePrompt] = useState({
    loading: true, 
    content: '',
    isCustom: false,
    isDirty: false
  });
  
  const [codebases, setCodebases] = useState({ default_codebases: [], user_codebases: [] });
  const [selectedCodebase, setSelectedCodebase] = useState(null); // This is for the CodebaseManager UI state
  const [newCodeText, setNewCodeText] = useState('');
  const [newCodeDescription, setNewCodeDescription] = useState('');
  const [newCodebaseName, setNewCodebaseName] = useState('');
  const [selectedBaseCodebase, setSelectedBaseCodebase] = useState('');
  
  useEffect(() => {
    const fetchFileStructure = async () => {
      try {
        setAnalyzing(true);
        const response = await axios.get(`${API_URL}/analyze-file/${file.filename}`);
        setFileStructure(response.data.structure);
        console.log("File structure:", response.data);
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
    
    const fetchFileContent = async () => {
      try {
        setLoadingContent(true);
        const response = await fetch(`${API_URL}/files/${file.filename}/content`);
        if (!response.ok) {
          throw new Error(`Failed to load file content: ${response.statusText}`);
        }
        const data = await response.json();
        setFileContent(data); // Assuming content is directly in data
        setLoadingContent(false);
      } catch (error) {
        console.error('Error loading file content:', error);
        setLoadingContent(false);
      }
    };

    fetchFileStructure();
    fetchFileContent();
    fetchCodebasesInternal(); // Renamed to avoid conflict if a prop fetchCodebases is passed
  }, [file.filename]);
  
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setConfig({
      ...config,
      [name]: type === 'checkbox' ? checked : 
              type === 'number' ? Number(value) :
              value
    });
  };
  
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
  
  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    const dynamicConfig = {
      ...config, // Spread all current config state
      file_id: file.filename, // Ensure file_id is correctly passed from prop
      // selected_codebase is already part of config state and updated by useEffect
    };
    console.log("Submitting configuration:", dynamicConfig);
    onSubmit(dynamicConfig, false); // false indicates not a standard config
  };
  
  const fetchPromptInternal = async (promptType) => {
    const targetSetter = promptType === 'inductive' ? setInductivePrompt : promptType === 'deductive' ? setDeductivePrompt : setParsePrompt;
    targetSetter(prev => ({ ...prev, loading: true }));
    try {
      const response = await axios.get(`${API_URL}/prompts/${promptType}`);
      targetSetter({
        content: response.data.content,
        isCustom: response.data.is_custom,
        loading: false,
        isDirty: false
      });
    } catch (err) {
      console.error(`Error fetching ${promptType} prompt:`, err);
      const defaultContent = promptType === 'inductive' 
        ? "Error loading inductive prompt. Using default prompt."
        : promptType === 'deductive'
          ? "Error loading deductive prompt. Using default prompt."
          : "Error loading parse prompt. Using default prompt.";
      targetSetter({
        content: defaultContent,
        isCustom: false,
        loading: false,
        isDirty: false
      });
    }
  };

  const savePromptInternal = async (promptType, content) => {
    const targetSetter = promptType === 'inductive' ? setInductivePrompt : promptType === 'deductive' ? setDeductivePrompt : setParsePrompt;
    try {
      const response = await axios.post(`${API_URL}/prompts/${promptType}`, content, {
        headers: { 'Content-Type': 'text/plain' }
      });
      targetSetter(prev => ({ 
        ...prev, 
        isCustom: response.data.is_custom,
        isDirty: false // Mark as not dirty after save
      }));
    } catch (err) {
      console.error(`Error saving ${promptType} prompt:`, err);
    }
  };

  const resetPromptInternal = async (promptType) => {
    try {
      await axios.delete(`${API_URL}/prompts/${promptType}`);
      fetchPromptInternal(promptType); // Fetch the default prompt after resetting
    } catch (err) {
      console.error(`Error resetting ${promptType} prompt:`, err);
    }
  };

  const fetchCodebasesInternal = async () => {
    try {
      // setCodebases(prev => ({ ...prev, loading: true })); // If you add a loading state for codebases list
      const response = await axios.get(`${API_URL}/codebases/list`);
      setCodebases(response.data);
      // setCodebases(prev => ({ ...prev, loading: false }));
    } catch (error) {
      console.error('Error fetching codebases:', error);
      // setCodebases(prev => ({ ...prev, loading: false }));
    }
  };

  const createCodebaseInternal = async () => {
    if (!newCodebaseName) {
      setError('Codebase name is required');
      return;
    }
    try {
      const formData = new FormData();
      formData.append('codebase_name', newCodebaseName);
      if (selectedBaseCodebase) {
        formData.append('base_codebase', selectedBaseCodebase);
      }
      await axios.post(`${API_URL}/codebases/create`, formData);
      setNewCodebaseName('');
      setSelectedBaseCodebase('');
      fetchCodebasesInternal();
    } catch (err) {
      console.error('Error creating codebase:', err);
      setError(`Failed to create codebase: ${err.response?.data?.detail || err.message}`);
    }
  };

  const addCodeToCodebaseInternal = async () => {
    if (!config.selected_codebase) { // Use config.selected_codebase for the target
      setError('Please select a codebase first');
      return;
    }
    if (!newCodeText) {
      setError('Code text is required');
      return;
    }
    try {
      await axios.post(`${API_URL}/codebases/${config.selected_codebase}/add_code`, {
        text: newCodeText,
        metadata: { description: newCodeDescription }
      });
      setNewCodeText('');
      setNewCodeDescription('');
      fetchCodebasesInternal(); // Refresh list
    } catch (err) {
      console.error('Error adding code:', err);
      setError(`Failed to add code: ${err.response?.data?.detail || err.message}`);
    }
  };

  const deleteCodebaseInternal = async (codebaseName) => {
    try {
      await axios.delete(`${API_URL}/codebases/${codebaseName}`);
      if (config.selected_codebase === codebaseName) {
        setConfig(prev => ({ ...prev, selected_codebase: '' })); // Clear from main form config
      }
      if (selectedCodebase === codebaseName) { // Also clear from CodebaseManager's own active selection
        setSelectedCodebase(null);
      }
      fetchCodebasesInternal();
    } catch (err) {
      console.error('Error deleting codebase:', err);
      setError(`Failed to delete codebase: ${err.response?.data?.detail || err.message}`);
    }
  };

  useEffect(() => {
    if (activeTab === 'inductive' && inductivePrompt.loading) {
      fetchPromptInternal('inductive');
    } else if (activeTab === 'deductive' && deductivePrompt.loading) {
      fetchPromptInternal('deductive');
    } else if (activeTab === 'parse' && parsePrompt.loading) {
      fetchPromptInternal('parse');
    }
    // Removed codebases.loading check as it's not explicitly set here
    // else if (activeTab === 'codebases') {
    //   fetchCodebasesInternal();
    // }
  }, [activeTab, inductivePrompt.loading, deductivePrompt.loading, parsePrompt.loading]);
  
  // Update main config when the CodebaseManager's selectedCodebase (for editing) changes
  // This ensures the dropdown in config tab stays in sync if user selects via CodebaseManager tab
  // useEffect(() => {
  //   if (selectedCodebase) { // selectedCodebase is the one from CodebaseManager for its own UI
  //     setConfig(prev => ({
  //       ...prev,
  //       selected_codebase: selectedCodebase // Update the main config as well
  //     }));
  //   }
  // }, [selectedCodebase]);
  // The above selectedCodebase is for the UI of CodebaseManager, config.selected_codebase is for the form submission.
  // The dropdown in the config tab directly sets config.selected_codebase.
  // The CodebaseManager uses its own `activeCodebase` for display, and `setSelectedCodebase` to update that.
  // For adding codes, it should use `config.selected_codebase` as the target.

  // Fetch codebases if deductive mode is selected and list is empty (initial load)
  useEffect(() => {
    if (config.coding_mode === 'deductive' && codebases.default_codebases.length === 0 && codebases.user_codebases.length === 0) {
      fetchCodebasesInternal();
    }
  }, [config.coding_mode]);
  
  const isFormValid = () => {
    if (config.coding_mode === 'deductive' && !config.selected_codebase) return false;
    if (!config.content_field && !file.is_default) return false; // Assuming file.is_default exists
    return true;
  };
  
  useEffect(() => {
    if (fileContent && config.content_field && previewTab === 'prompt') {
      generatePromptPreview();
    }
  }, [
    config.content_field, config.context_fields, config.list_field, 
    config.coding_mode, config.selected_codebase, config.meaning_units_per_assignment_prompt,
    config.preliminary_segments_per_prompt, config.use_parsing, config.context_size,
    fileContent, file.filename, previewTab // Added previewTab
  ]);
  
  const generatePromptPreview = () => {
    if (!fileContent || !config.content_field) {
      setPromptPreview("Configure required fields to see prompt preview.");
      return;
    }
    setPromptPreview("Loading prompt preview...");
    const previewConfig = {
      file_id: file.filename,
      ...config, // Send all current config settings
    };
    axios.post(`${API_URL}/preview-prompt`, previewConfig)
      .then(response => setPromptPreview(response.data.prompt))
      .catch(error => {
        console.error("Error fetching prompt preview:", error);
        setPromptPreview("Error fetching prompt preview. Ensure settings are correct.");
      });
  };
  
  if (analyzing && !fileStructure) {
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
      <p className="help-text">Set up how your file should be processed.</p>
      
      <div className="config-tabs">
        {['config', 'inductive', 'deductive', 'parse', 'codebases', 'go_back'].map(tabName => (
          <button 
            key={tabName}
            className={`tab-button ${activeTab === tabName ? 'active' : ''}`}
            onClick={() => tabName === 'go_back' ? onCancel() : setActiveTab(tabName)}
          >
            {tabName === 'go_back' ? 'Go Back' : tabName.charAt(0).toUpperCase() + tabName.slice(1)}
          </button>
        ))}
      </div>
      
      {activeTab === 'config' && (
        <>
          {/* File Mapping and Options Sections */}
          <div className="file-config-layout">
            <div className="file-mapping-section">
              <div className="form-section">
                <h3>Field Mapping</h3>
                <div className="form-group">
                  <label htmlFor="content-field">Content Field:</label>
                  <select 
                    id="content-field" 
                    name="content_field" // Added name attribute
                    value={config.content_field}
                    onChange={handleChange} // Use general handleChange
                    required
                  >
                    <option value="">-- Select Field --</option>
                    {fileStructure?.fields?.map(field => (
                      <option key={field.name} value={field.name}>{field.name}</option>
                    ))}
                  </select>
                  <small>Main text content to analyze</small>
                </div>
                
                <div className="form-group">
                  <label>Context Fields:</label>
                  <div className="checkbox-group">
                    {fileStructure?.fields?.map(field => (
                      <label key={field.name} className="checkbox-label">
                        <input 
                          type="checkbox"
                          checked={config.context_fields.includes(field.name)}
                          onChange={e => handleContextFieldChange(field.name, e.target.checked)} // Use specific handler
                        />
                        {field.name}
                      </label>
                    ))}
                  </div>
                  <small>Additional fields for context</small>
                </div>
                
                {/* {(fileStructure?.arrays?.length > 0 || fileStructure?.fields?.some(f => f.type === 'array')) && ( */} 
                {/* Simplified to show if any field can be a list. Backend determines if it's valid. */} 
                <div className="form-group">
                  <label htmlFor="list-field">List/Nested Data Field:</label>
                  <select 
                    id="list-field" 
                    name="list_field" // Added name
                    value={config.list_field}
                    onChange={handleChange} // Use general handleChange
                  >
                    <option value="">-- None (e.g. root is list or no list) --</option>
                    {fileStructure?.fields?.map(field => (
                      <option key={field.name} value={field.name}>{field.name}</option>
                    ))}
                  </select>
                  <small>Field containing an array of items to process (leave empty if root is array or N/A)</small>
                </div>
                {/* )} */} 
              </div>
              
              <div className="form-section">
                <h3>Coding Options</h3>
                <div className="form-group">
                  <label htmlFor="coding-mode">Coding Mode:</label>
                  <select 
                    id="coding-mode" 
                    name="coding_mode" // Added name
                    value={config.coding_mode}
                    onChange={handleChange} // Use general handleChange
                  >
                    <option value="inductive">Inductive (generate codes)</option>
                    <option value="deductive">Deductive (use predefined codes)</option>
                  </select>
                </div>
                
                {config.coding_mode === 'deductive' && (
                  <div className="form-group">
                    <label htmlFor="selected-codebase">Select Codebase:</label>
                    <select 
                      id="selected-codebase" 
                      name="selected_codebase" // Added name
                      value={config.selected_codebase}
                      onChange={handleChange} // Use general handleChange
                      required
                    >
                      <option value="">-- Select Codebase --</option>
                      {codebases.default_codebases?.map(cb => (
                        <option key={cb.filename} value={cb.filename}>{cb.filename} (Default)</option>
                      ))}
                      {codebases.user_codebases?.map(cb => (
                        <option key={cb.filename} value={cb.filename}>{cb.filename}</option>
                      ))}
                    </select>
                    {!config.selected_codebase && (
                      <span className="validation-message">Codebase required for deductive coding</span>
                    )}
                  </div>
                )}
                
                <div className="form-group">
                  <label className="checkbox-label">
                    <input 
                      type="checkbox"
                      name="use_parsing" // Added name
                      checked={config.use_parsing}
                      onChange={handleChange} // Use general handleChange
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
                    <label htmlFor="model_name">Model:</label>
                    <select 
                      id="model_name" 
                      name="model_name" // Name matches state key
                      value={config.model_name}
                      onChange={handleChange}
                    >
                      <option value="gpt-4o-mini">GPT-4o-mini</option>
                      <option value="gpt-4o">GPT-4o</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label htmlFor="temperature">Temperature:</label>
                    <input type="number" id="temperature" name="temperature" min="0" max="2" step="0.1" value={config.temperature} onChange={handleChange} />
                  </div>
                  <div className="form-group">
                    <label htmlFor="max_tokens">Max Tokens:</label>
                    <input type="number" id="max_tokens" name="max_tokens" min="100" max="4000" step="100" value={config.max_tokens} onChange={handleChange} />
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label htmlFor="thread_count">Thread Count:</label>
                    <input type="number" id="thread_count" name="thread_count" min="1" max="10" value={config.thread_count} onChange={handleChange} />
                  </div>
                  <div className="form-group">
                    <label htmlFor="preliminary_segments_per_prompt">Segments Per Parsing Prompt:</label>
                    <input type="number" id="preliminary_segments_per_prompt" name="preliminary_segments_per_prompt" min="1" max="10" value={config.preliminary_segments_per_prompt} onChange={handleChange} />
                  </div>
                  <div className="form-group">
                    <label htmlFor="meaning_units_per_assignment_prompt">Meaning Units Per Assignment:</label>
                    <input type="number" id="meaning_units_per_assignment_prompt" name="meaning_units_per_assignment_prompt" min="1" max="20" value={config.meaning_units_per_assignment_prompt} onChange={handleChange} />
                  </div>
                  <div className="form-group">
                    <label htmlFor="context_size">Context Window Size:</label>
                    <input type="number" id="context_size" name="context_size" min="1" max="20" value={config.context_size} onChange={handleChange} />
                  </div>
                </div>
              </div>
            </div>

            {/* Preview Section */}
            <div className="file-preview-section">
              <div className="preview-tabs">
                <button className={`tab-button ${previewTab === 'file' ? 'active' : ''}`} onClick={() => setPreviewTab('file')}>File Preview</button>
                <button className={`tab-button ${previewTab === 'prompt' ? 'active' : ''}`} onClick={() => setPreviewTab('prompt')}>Prompt Preview</button>
              </div>
              <div className="json-document-container">
                {loadingContent && previewTab === 'file' ? (
                  <div className="loading-content">Loading file content...</div>
                ) : previewTab === 'file' ? (
                  <ExpandableContainer title="File Preview">
                    <JsonDocumentViewer content={fileContent} />
                  </ExpandableContainer>
                ) : (
                  <ExpandableContainer title="Prompt Preview">
                    <div className="prompt-preview">
                      <pre>{promptPreview}</pre>
                    </div>
                  </ExpandableContainer>
                )}
              </div>
            </div>
          </div>
          
          <div className="form-actions">
            <button className="secondary-button" onClick={onCancel}>Cancel</button>
            <button className="primary-button" onClick={handleSubmit} disabled={!isFormValid()}>Start Processing</button>
          </div>
        </>
      )}
      
      {activeTab === 'inductive' && (
        <PromptEditor 
          type="inductive"
          prompt={inductivePrompt}
          setPrompt={setInductivePrompt}
          fetchPrompt={fetchPromptInternal} // Pass internal versions
          savePrompt={savePromptInternal}
          resetPrompt={resetPromptInternal}
        />
      )}
      
      {activeTab === 'deductive' && (
        <PromptEditor 
          type="deductive"
          prompt={deductivePrompt}
          setPrompt={setDeductivePrompt}
          fetchPrompt={fetchPromptInternal}
          savePrompt={savePromptInternal}
          resetPrompt={resetPromptInternal}
        />
      )}
      
      {activeTab === 'parse' && (
        <PromptEditor 
          type="parse"
          prompt={parsePrompt}
          setPrompt={setParsePrompt}
          fetchPrompt={fetchPromptInternal}
          savePrompt={savePromptInternal}
          resetPrompt={resetPromptInternal}
        />
      )}
      
      {activeTab === 'codebases' && (
        <CodebaseManager 
          codebases={codebases} // Pass full codebases object
          selectedCodebase={selectedCodebase} // UI selection state for CodebaseManager
          setSelectedCodebase={setSelectedCodebase} // To update its own selection
          newCodeText={newCodeText}
          setNewCodeText={setNewCodeText}
          newCodeDescription={newCodeDescription}
          setNewCodeDescription={setNewCodeDescription}
          newCodebaseName={newCodebaseName}
          setNewCodebaseName={setNewCodebaseName}
          selectedBaseCodebase={selectedBaseCodebase}
          setSelectedBaseCodebase={setSelectedBaseCodebase}
          handleCreateCodebase={createCodebaseInternal}
          handleAddCode={addCodeToCodebaseInternal} // Pass internal version
          handleSelectCodebase={deleteCodebaseInternal} // Pass internal version (delete)
          fetchCodebases={fetchCodebasesInternal} // Pass internal version
        />
      )}
    </div>
  );
}

export default FileConfigForm; 