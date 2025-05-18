import React from 'react';
import ExpandableContainer from './ExpandableContainer'; // Updated import path
import { API_URL } from '../config'; // Added API_URL import
import axios from 'axios'; // Added axios import

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
      
      <ExpandableContainer title={`${type.charAt(0).toUpperCase() + type.slice(1)} Prompt Editor`}>
        <textarea 
          className="prompt-textarea"
          value={prompt.content || ''}
          onChange={handlePromptChange}
          rows={15}
        />
      </ExpandableContainer>
      
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

export default PromptEditor; 