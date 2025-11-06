import React from 'react';

function Message({ message }) {
  const { role, content, metadata, isError } = message;

  return (
    <div className={`message ${role}-message ${isError ? 'error-message' : ''}`}>
      <div className="message-content">
        <p className="message-text">{content}</p>
        
        {metadata && (
          <div className="message-metadata">
            <span className="metadata-badge" title="Intent">
              {metadata.intent}
            </span>
            <span className="metadata-badge" title="Confidence">
              {(metadata.confidence * 100).toFixed(0)}% confident
            </span>
            <span className="metadata-badge" title="Quality Score">
              Q: {(metadata.quality_score * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export default Message;