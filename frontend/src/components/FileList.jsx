import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faFileCode, faTrash } from '@fortawesome/free-solid-svg-icons';
import JsonDocumentViewer from './JsonDocumentViewer';
import ExpandableContainer from './ExpandableContainer';
import { API_URL } from '../config';

function FileList({ files, onSelect, selectedFile, onDelete }) {
  const [viewingContent, setViewingContent] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSelect = async (file) => {
    onSelect(file);
    
    // Load file content for preview
    try {
      setLoading(true);
      // Using fetch as in the original code, could be changed to axios for consistency
      const response = await fetch(`${API_URL}/files/${file.filename}/content`);
      if (!response.ok) {
        throw new Error(`Failed to load file content: ${response.statusText}`);
      }
      const data = await response.json();
      // Assuming the content is directly in data, not data.content based on other similar fetches
      setViewingContent(data);
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
            <ExpandableContainer title="File Preview">
              <JsonDocumentViewer content={viewingContent} />
            </ExpandableContainer>
          )}
        </div>
      </div>
    </div>
  );
}

export default FileList; 