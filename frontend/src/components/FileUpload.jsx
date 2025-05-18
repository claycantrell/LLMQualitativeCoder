import React, { useState, useRef } from 'react';
import axios from 'axios';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faUpload } from '@fortawesome/free-solid-svg-icons';
import { API_URL } from '../config';

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

export default FileUpload; 