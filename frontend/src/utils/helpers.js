import { API_URL } from '../config';

// Helper function to format dates
export const formatDate = (dateString) => {
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
export const downloadOutput = (jobId) => {
  window.open(`${API_URL}/jobs/${jobId}/output`, '_blank');
};

// Helper function to download validation
export const downloadValidation = (jobId) => {
  window.open(`${API_URL}/jobs/${jobId}/validation`, '_blank');
}; 