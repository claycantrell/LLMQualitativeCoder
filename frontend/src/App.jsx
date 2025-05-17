import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

// API base URL - change this to your API URL when deployed
const API_URL = 'http://localhost:8000'

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
    thread_count: 2
  })
  
  // State for jobs and selected job
  const [jobs, setJobs] = useState([])
  const [selectedJob, setSelectedJob] = useState(null)
  const [selectedJobData, setSelectedJobData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  
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
    setLoading(true)
    setError(null)
    
    try {
      const response = await axios.post(`${API_URL}/run-pipeline`, config)
      fetchJobs() // Refresh job list
      setSelectedJob(response.data.job_id)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start pipeline')
      console.error('Error starting pipeline:', err)
    } finally {
      setLoading(false)
    }
  }
  
  // Function to fetch all jobs
  const fetchJobs = async () => {
    try {
      const response = await axios.get(`${API_URL}/jobs`)
      setJobs(response.data.jobs)
    } catch (err) {
      console.error('Error fetching jobs:', err)
    }
  }
  
  // Function to fetch job details
  const fetchJobDetails = async (jobId) => {
    if (!jobId) return
    
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
  
  // Fetch jobs on component mount
  useEffect(() => {
    fetchJobs()
    
    // Set up polling for jobs list
    const interval = setInterval(fetchJobs, 5000)
    return () => clearInterval(interval)
  }, [])
  
  // Fetch job details when selected job changes
  useEffect(() => {
    if (selectedJob) {
      fetchJobDetails(selectedJob)
    }
  }, [selectedJob])
  
  return (
    <div className="container">
      <header>
        <h1>LLM Qualitative Coder</h1>
        <p>Automated qualitative coding using Large Language Models</p>
      </header>
      
      <div className="content">
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
            
            <div className="form-group">
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
              />
            </div>
            
            <div className="form-group">
              <label>Meaning Units Per Prompt:</label>
              <input 
                type="number" 
                name="meaning_units_per_assignment_prompt" 
                value={config.meaning_units_per_assignment_prompt} 
                onChange={handleChange}
              />
            </div>
            
            <div className="form-group">
              <label>Context Size:</label>
              <input 
                type="number" 
                name="context_size" 
                value={config.context_size} 
                onChange={handleChange}
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
              <button type="submit" disabled={loading}>
                {loading ? 'Starting Pipeline...' : 'Start Pipeline'}
              </button>
            </div>
          </form>
        </div>
        
        <div className="jobs-panel">
          <h2>Jobs</h2>
          <div className="jobs-list">
            {jobs.length === 0 ? (
              <p>No jobs yet. Start a new pipeline.</p>
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
                      <div>Job ID: {job.job_id.substring(0, 8)}...</div>
                      <div>Status: {job.status}</div>
                      <div>Started: {new Date(job.started_at).toLocaleString()}</div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
          
          {selectedJobData && (
            <div className="job-details">
              <h3>Job Details</h3>
              <div>Status: {selectedJobData.status}</div>
              <div>Started: {new Date(selectedJobData.started_at).toLocaleString()}</div>
              
              {selectedJobData.completed_at && (
                <div>Completed: {new Date(selectedJobData.completed_at).toLocaleString()}</div>
              )}
              
              {selectedJobData.error && (
                <div className="error">Error: {selectedJobData.error}</div>
              )}
              
              {selectedJobData.status === 'completed' && (
                <div className="job-actions">
                  <button onClick={() => downloadOutput(selectedJobData.job_id)}>
                    Download Output
                  </button>
                  <button onClick={() => downloadValidation(selectedJobData.job_id)}>
                    Download Validation
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
