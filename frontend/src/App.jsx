import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

import { API_URL } from './config';
import { formatDate, downloadOutput, downloadValidation } from './utils/helpers';

// Import moved components
import FileConfigForm from './components/FileConfigForm';
import FileList from './components/FileList';
import FileUpload from './components/FileUpload';

function App() {
  const [files, setFiles] = useState({ user_files: [], default_files: [] });
  const [selectedFile, setSelectedFile] = useState(null);
  const [showFileConfig, setShowFileConfig] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('');
  const [jobs, setJobs] = useState([]);
  const [activeJobId, setActiveJobId] = useState(null);
  
  useEffect(() => {
    fetchFiles();
    const intervalId = setInterval(fetchJobStatus, 5000);
    return () => clearInterval(intervalId);
  }, []);

  const fetchFiles = async () => {
    try {
      const response = await axios.get(`${API_URL}/files/list`);
      setFiles(response.data);
    } catch (error) {
      console.error('Error fetching files:', error);
    }
  };

  const fetchJobStatus = async () => {
    try {
      const response = await axios.get(`${API_URL}/jobs`);
      setJobs(response.data.jobs || []);
    } catch (error) {
      console.error('Error fetching job status:', error);
    }
  };

  const handleFileSelect = (file) => {
    setSelectedFile(file);
  };

  const handleDeleteFile = async (filename) => {
    if (window.confirm(`Are you sure you want to delete ${filename}?`)) {
      try {
        await axios.delete(`${API_URL}/files/${filename}`);
        fetchFiles();
        if (selectedFile?.filename === filename) {
          setSelectedFile(null);
        }
      } catch (error) {
        console.error('Error deleting file:', error);
        alert(`Failed to delete file: ${error.response?.data?.detail || error.message}`);
      }
    }
  };

  const configureProcessingForFile = (fileToConfigure) => {
    setSelectedFile(fileToConfigure);
    setShowFileConfig(true);
  };

  const handleConfigSubmit = async (configData) => {
    try {
      const response = await axios.post(`${API_URL}/run-pipeline-with-config`, configData);
      setShowFileConfig(false);
      setSelectedFile(null);
      fetchJobStatus(); 
      setActiveJobId(response.data.job_id);
      alert(`Processing started for job ID: ${response.data.job_id}`)
    } catch (error) {
      console.error('Error submitting configuration:', error);
      alert(`Failed to start processing: ${error.response?.data?.detail || error.message}`);
    }
  };

  const handleCancelConfig = () => {
    setShowFileConfig(false);
    setSelectedFile(null);
  };

  return (
    <div className="App">
      <div className="content-wrapper">
        <header className="App-header">
          <h1>LLM Qualitative Coder</h1>
          <p>Automated qualitative coding using Large Language Models</p>
        </header>
        
        <div className="content">
          {showFileConfig && selectedFile ? (
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
                
                {uploadStatus && (
                    <div className="upload-status">
                        {uploadProgress > 0 && uploadProgress < 100 && (
                        <div className="progress-bar">
                            <div className="progress" style={{width: `${uploadProgress}%`}}></div>
                        </div>
                        )}
                        <p className="status-message">{uploadStatus}</p>
                    </div>
                )}
                
                <FileList 
                  files={files}
                  onSelect={handleFileSelect}
                  selectedFile={selectedFile}
                  onDelete={handleDeleteFile}
                />
                
                {selectedFile && !showFileConfig && (
                  <div className="file-actions">
                    <button 
                        className="primary-button" 
                        onClick={() => configureProcessingForFile(selectedFile)}
                    >
                      Configure Processing
                    </button>
                  </div>
                )}
              </section>

              <section className="jobs-section">
                <h2>Processing Jobs</h2>
                {jobs.length === 0 ? (
                  <p className="empty-message">No jobs yet. Configure and start processing a file.</p>
                ) : (
                  <ul className="jobs-list">
                    {jobs.map(job => (
                      <li 
                        key={job.job_id}
                        className={activeJobId === job.job_id ? 'selected' : ''}
                        onClick={() => setActiveJobId(job.job_id)}
                      >
                        <div className={`job-status ${job.status?.toLowerCase()}`}></div>
                        <div className="job-info">
                          <div className="job-name">{job.config?.file_id || job.filename || 'Job ' + job.job_id.substring(0,8)}</div>
                          <div className="job-meta">
                            <span className="job-date">Started: {formatDate(job.started_at)}</span>
                            <span className={`job-status-text ${job.status?.toLowerCase()}`}>{job.status}</span>
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
                
                {activeJobId && jobs.find(j => j.job_id === activeJobId) && (() => {
                    const job = jobs.find(j => j.job_id === activeJobId);
                    if (!job) return null;
                    return (
                        <div className="job-details">
                        <h3>Job Details: {job.config?.file_id || job.filename || job.job_id.substring(0,8)}</h3>
                        <div><strong>Status:</strong> <span className={`job-status-text ${job.status?.toLowerCase()}`}>{job.status}</span></div>
                        <div><strong>Job ID:</strong> {job.job_id}</div>
                        <div><strong>Started:</strong> {formatDate(job.started_at)}</div>
                        {job.completed_at && (
                            <div><strong>Completed:</strong> {formatDate(job.completed_at)}</div>
                        )}
                        {job.error && (
                            <div className="error"><strong>Error:</strong> {job.error}</div>
                        )}
                        {job.status === 'completed' && (
                            <div className="job-actions">
                            <button className="secondary-button" onClick={() => downloadOutput(activeJobId)}>
                                Download Output
                            </button>
                            <button className="secondary-button" onClick={() => downloadValidation(activeJobId)}>
                                Download Validation
                            </button>
                            </div>
                        )}
                        </div>
                    );
                })()}
              </section>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App
