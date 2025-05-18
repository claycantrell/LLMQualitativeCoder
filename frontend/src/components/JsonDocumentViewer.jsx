import React from 'react';

function JsonDocumentViewer({ content }) {
  if (!content) return <div className="json-document-empty">Select a file to view its content</div>;
  
  return (
    <div className="json-document-viewer">
      <pre>{JSON.stringify(content, null, 2)}</pre>
    </div>
  );
}

export default JsonDocumentViewer; 