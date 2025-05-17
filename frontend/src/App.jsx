import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

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

function App() {
  // State for config form
  const [config, setConfig] = useState({
    coding_mode: 'inductive',
    use_parsing: true,
    preliminary_segments_per_prompt: 50,
    meaning_units_per_assignment_prompt: 10,
    context_size: 5,
    data_format: 'transcript',
    model_name: 'gpt-4o-mini',
    temperature: 0.7,
    max_tokens: 2000,
    thread_count: 2,
    input_file: null  // Track the selected input file
  })
  
  // State for jobs and selected job
  const [jobs, setJobs] = useState([])
  const [selectedJob, setSelectedJob] = useState(null)
  const [selectedJobData, setSelectedJobData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [apiConnected, setApiConnected] = useState(false)
  
  // State for file management
  const [availableFiles, setAvailableFiles] = useState({
    default_files: [],
    user_files: []
  })
  const [uploadingFile, setUploadingFile] = useState(false)
  const [fileError, setFileError] = useState(null)
  
  // Check API connection on load
  useEffect(() => {
    const checkApiConnection = async () => {
      try {
        await axios.get(`${API_URL}/`)
        setApiConnected(true)
        setError(null)
      } catch (err) {
        console.error('API connection failed:', err)
        setError('Cannot connect to the API server. Please ensure it is running.')
        setApiConnected(false)
      }
    }
    
    checkApiConnection()
    const interval = setInterval(checkApiConnection, 5000)
    return () => clearInterval(interval)
  }, [])
  
  // Load available files on API connection
  useEffect(() => {
    if (apiConnected) {
      fetchAvailableFiles()
    }
  }, [apiConnected])
  
  // Fetch available files
  const fetchAvailableFiles = async () => {
    try {
      const response = await axios.get(`${API_URL}/files/list`)
      setAvailableFiles(response.data)
      
      // If no file is selected and there are files available, select the first default file
      if (!config.input_file && response.data.default_files.length > 0) {
        setConfig(prev => ({
          ...prev,
          input_file: response.data.default_files[0].filename
        }))
      }
    } catch (err) {
      console.error('Error fetching files:', err)
      setFileError('Failed to load available files')
    }
  }
  
  // Handle file upload
  const handleFileUpload = async (event) => {
    const file = event.target.files[0]
    if (!file) return
    
    // Validate file type
    if (!file.name.endsWith('.json')) {
      setFileError('Only JSON files are allowed')
      return
    }
    
    setUploadingFile(true)
    setFileError(null)
    
    try {
      const formData = new FormData()
      formData.append('file', file)
      
      const response = await axios.post(`${API_URL}/files/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      
      // Refresh file list and select the newly uploaded file
      await fetchAvailableFiles()
      setConfig(prev => ({
        ...prev,
        input_file: response.data.filename
      }))
    } catch (err) {
      console.error('Error uploading file:', err)
      setFileError(err.response?.data?.detail || 'Failed to upload file')
    } finally {
      setUploadingFile(false)
      // Clear file input
      event.target.value = null
    }
  }
  
  // Handle file deletion
  const handleDeleteFile = async (filename) => {
    try {
      await axios.delete(`${API_URL}/files/${filename}`)
      
      // If the deleted file was selected, reset selection
      if (config.input_file === filename) {
        setConfig(prev => ({
          ...prev,
          input_file: availableFiles.default_files.length > 0 ? availableFiles.default_files[0].filename : null
        }))
      }
      
      // Refresh file list
      fetchAvailableFiles()
    } catch (err) {
      console.error('Error deleting file:', err)
      setFileError(err.response?.data?.detail || 'Failed to delete file')
    }
  }
  
  // Function to handle form changes
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target
    setConfig({
      ...config,
      [name]: type === 'checkbox' ? checked : 
              type === 'number' ? Number(value) :
              value
    })
  }
  
  // Function to start a new pipeline
  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!apiConnected) {
      setError('Cannot start pipeline: API server is not connected')
      return
    }
    
    if (!config.input_file) {
      setError('Please select an input file before starting the pipeline')
      return
    }
    
    setLoading(true)
    setError(null)
    
    try {
      // Debug the request
      console.log('Sending request to:', `${API_URL}/run-pipeline`)
      console.log('With payload:', config)
      
      const response = await axios.post(`${API_URL}/run-pipeline`, config, {
        headers: {
          'Content-Type': 'application/json'
        }
      })
      
      console.log('Response:', response.data)
      fetchJobs() // Refresh job list
      setSelectedJob(response.data.job_id)
    } catch (err) {
      console.error('Error starting pipeline:', err)
      if (err.response) {
        console.error('Response data:', err.response.data)
        console.error('Response status:', err.response.status)
        setError(`Server error (${err.response.status}): ${err.response?.data?.detail || 'Unknown error'}`)
      } else if (err.request) {
        setError('No response from server. Please check if the API is running.')
      } else {
        setError(`Error: ${err.message}`)
      }
    } finally {
      setLoading(false)
    }
  }
  
  // Function to fetch all jobs
  const fetchJobs = async () => {
    if (!apiConnected) return
    
    try {
      const response = await axios.get(`${API_URL}/jobs`)
      setJobs(response.data.jobs || [])
    } catch (err) {
      console.error('Error fetching jobs:', err)
    }
  }
  
  // Function to fetch job details
  const fetchJobDetails = async (jobId) => {
    if (!jobId || !apiConnected) return
    
    try {
      const response = await axios.get(`${API_URL}/jobs/${jobId}`)
      setSelectedJobData(response.data)
      
      // Auto-refresh if job is still processing
      if (response.data.status === 'pending' || response.data.status === 'running') {
        setTimeout(() => fetchJobDetails(jobId), 2000)
      }
    } catch (err) {
      console.error('Error fetching job details:', err)
    }
  }
  
  // Function to download output file
  const downloadOutput = async (jobId) => {
    try {
      window.open(`${API_URL}/jobs/${jobId}/output`, '_blank')
    } catch (err) {
      console.error('Error downloading output:', err)
    }
  }
  
  // Function to download validation file
  const downloadValidation = async (jobId) => {
    try {
      window.open(`${API_URL}/jobs/${jobId}/validation`, '_blank')
    } catch (err) {
      console.error('Error downloading validation:', err)
    }
  }
  
  // Fetch jobs on component mount and when API connection changes
  useEffect(() => {
    if (apiConnected) {
      fetchJobs()
      const interval = setInterval(fetchJobs, 5000)
      return () => clearInterval(interval)
    }
  }, [apiConnected])
  
  // Fetch job details when selected job changes
  useEffect(() => {
    if (selectedJob) {
      fetchJobDetails(selectedJob)
    }
  }, [selectedJob])
  
  // Format date for readable display
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A'
    
    const date = new Date(dateString)
    return date.toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  }
  
  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }
  
  return (
    <div className="container">
      <header>
        <h1>LLM Qualitative Coder</h1>
        <p>Automated qualitative coding using Large Language Models</p>
        <div className={`api-status ${apiConnected ? 'connected' : 'disconnected'}`}>
          API Status: {apiConnected ? 'Connected' : 'Disconnected'}
        </div>
      </header>
      
      <div className="content">
        <div className="config-panel">
          {/* File Selection Panel */}
          <div className="file-panel">
            <h2>Input Files</h2>
            {fileError && <div className="error">{fileError}</div>}
            
            <div className="file-upload">
              <label className="upload-button">
                <UploadIcon /> Upload JSON File
                <input 
                  type="file" 
                  accept=".json" 
                  onChange={handleFileUpload} 
                  disabled={uploadingFile || !apiConnected}
                  style={{ display: 'none' }}
                />
              </label>
              {uploadingFile && <span className="upload-status">Uploading...</span>}
            </div>
            
            <div className="file-list">
              <h3>Default Files</h3>
              <ul>
                {availableFiles.default_files.map(file => (
                  <li 
                    key={file.filename}
                    className={config.input_file === file.filename ? 'selected' : ''}
                  >
                    <label className="file-item">
                      <input 
                        type="radio" 
                        name="input_file" 
                        value={file.filename}
                        checked={config.input_file === file.filename}
                        onChange={handleChange}
                      />
                      <div className="file-info">
                        <div className="file-name">{file.filename}</div>
                        <div className="file-meta">
                          <span>{formatFileSize(file.size)}</span>
                          <span className="file-default">Default</span>
                        </div>
                      </div>
                    </label>
                  </li>
                ))}
              </ul>
              
              <h3>User Files</h3>
              <ul>
                {availableFiles.user_files.length === 0 ? (
                  <li className="empty-list">No user files uploaded</li>
                ) : (
                  availableFiles.user_files.map(file => (
                    <li 
                      key={file.filename}
                      className={config.input_file === file.filename ? 'selected' : ''}
                    >
                      <label className="file-item">
                        <input 
                          type="radio" 
                          name="input_file" 
                          value={file.filename}
                          checked={config.input_file === file.filename}
                          onChange={handleChange}
                        />
                        <div className="file-info">
                          <div className="file-name">{file.filename}</div>
                          <div className="file-meta">
                            <span>{formatFileSize(file.size)}</span>
                            <span>{new Date(file.upload_date).toLocaleDateString()}</span>
                          </div>
                        </div>
                      </label>
                      <button 
                        className="delete-button"
                        onClick={() => handleDeleteFile(file.filename)}
                        title="Delete file"
                      >
                        <TrashIcon />
                      </button>
                    </li>
                  ))
                )}
              </ul>
            </div>
          </div>
          
          {/* Configuration Panel */}
          <div className="config-form">
            <h2>Pipeline Configuration</h2>
            {error && <div className="error">{error}</div>}
            
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label>Coding Mode:</label>
                <select 
                  name="coding_mode" 
                  value={config.coding_mode} 
                  onChange={handleChange}
                >
                  <option value="inductive">Inductive</option>
                  <option value="deductive">Deductive</option>
                </select>
              </div>
              
              <div className="form-group checkbox-group">
                <label>
                  <input 
                    type="checkbox" 
                    name="use_parsing" 
                    checked={config.use_parsing} 
                    onChange={handleChange}
                  />
                  Use Parsing
                </label>
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
                <label>Meaning Units Per Prompt:</label>
                <input 
                  type="number" 
                  name="meaning_units_per_assignment_prompt" 
                  value={config.meaning_units_per_assignment_prompt} 
                  onChange={handleChange}
                  min="1"
                  max="50"
                />
              </div>
              
              <div className="form-group">
                <label>Context Size:</label>
                <input 
                  type="number" 
                  name="context_size" 
                  value={config.context_size} 
                  onChange={handleChange}
                  min="1"
                  max="20"
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
              
              <div className="form-group">
                <label>Threads:</label>
                <input 
                  type="number" 
                  name="thread_count" 
                  value={config.thread_count} 
                  onChange={handleChange}
                  min="1"
                  max="10"
                />
              </div>
              
              <div className="form-actions">
                <button 
                  type="submit" 
                  disabled={loading || !apiConnected || !config.input_file} 
                  className="start-button"
                >
                  {loading ? 'Starting Pipeline...' : 'Start Pipeline'}
                </button>
              </div>
            </form>
          </div>
        </div>
        
        <div className="jobs-panel">
          <h2>Jobs</h2>
          <div className="jobs-list">
            {jobs.length === 0 ? (
              <p>{apiConnected ? 'No jobs yet. Start a new pipeline.' : 'Connect to API to view jobs'}</p>
            ) : (
              <ul>
                {jobs.map(job => (
                  <li 
                    key={job.job_id}
                    className={selectedJob === job.job_id ? 'selected' : ''}
                    onClick={() => setSelectedJob(job.job_id)}
                  >
                    <div className={`job-status ${job.status}`}></div>
                    <div className="job-info">
                      <div><strong>ID:</strong> {job.job_id.substring(0, 8)}...</div>
                      <div><strong>Status:</strong> {job.status}</div>
                      <div><strong>Started:</strong> {formatDate(job.started_at)}</div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
          
          {selectedJobData && (
            <div className="job-details">
              <h3>Job Details</h3>
              <div><strong>Status:</strong> {selectedJobData.status}</div>
              <div><strong>Started:</strong> {formatDate(selectedJobData.started_at)}</div>
              
              {selectedJobData.completed_at && (
                <div><strong>Completed:</strong> {formatDate(selectedJobData.completed_at)}</div>
              )}
              
              {selectedJobData.error && (
                <div className="error">Error: {selectedJobData.error}</div>
              )}
              
              {selectedJobData.status === 'completed' && (
                <div className="job-actions">
                  <button onClick={() => downloadOutput(selectedJobData.job_id)}>
                    <DownloadIcon /> Download Output
                  </button>
                  <button onClick={() => downloadValidation(selectedJobData.job_id)}>
                    <DocumentIcon /> Download Validation
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
