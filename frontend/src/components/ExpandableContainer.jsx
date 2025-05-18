import React, { useState, useEffect, useRef } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faExpand, faCompress } from '@fortawesome/free-solid-svg-icons';

function ExpandableContainer({ children, title }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const containerRef = useRef(null);

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  // Add event listener to close on escape key
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && isExpanded) {
        setIsExpanded(false);
      }
    };

    if (isExpanded) {
      document.addEventListener('keydown', handleEscape);
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isExpanded]);

  return (
    <div 
      ref={containerRef}
      className={`expandable-container ${isExpanded ? 'expanded' : ''}`}
    >
      <div className="expandable-header">
        {title && <h3>{title}</h3>}
        <button 
          className="expand-button"
          onClick={toggleExpand}
          title={isExpanded ? "Minimize" : "Maximize"}
        >
          <FontAwesomeIcon icon={isExpanded ? faCompress : faExpand} />
        </button>
      </div>
      <div className="expandable-content">
        {children}
      </div>
    </div>
  );
}

export default ExpandableContainer; 