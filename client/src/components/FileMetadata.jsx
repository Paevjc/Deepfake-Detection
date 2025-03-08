import React from 'react';

/**
 * FileMetadata Component
 * 
 * Displays metadata information for an uploaded file.
 * 
 * @component
 * @param {Object} props - Component props
 * @param {Object} props.metadata - File metadata object containing file information
 * @param {string} props.metadata.name - Name of the file
 * @param {string} props.metadata.size - Formatted size of the file (e.g., "2.5 MB")
 * @param {string} props.metadata.type - MIME type of the file
 * @param {string} [props.metadata.dimensions] - Optional dimensions of the file (for images/videos)
 * @param {string} [props.metadata.duration] - Optional duration of the file (for videos)
 * @returns {JSX.Element|null} - Returns the file metadata display or null if no metadata
 */

const FileMetadata = ({ metadata }) => {
  if (!metadata) return null;

  return (
    <div className="mb-6">
      <h3 className="text-xl text-black font-semibold text-center mb-4">File Information</h3>
      <div className="bg-white text-black p-6 rounded-lg border-2 border-solid border-gray-200 text-left">
        <p className="font-mono text-lg">File name: {metadata.name}</p>
        <p className="font-mono text-lg mt-2">Size: {metadata.size}</p>
        <p className="font-mono text-lg mt-2">Type: {metadata.type}</p>
        {metadata.dimensions && (
          <p className="font-mono text-lg mt-2">Resolution: {metadata.dimensions}</p>
        )}
        {metadata.duration && (
          <p className="font-mono text-lg mt-2">Duration: {metadata.duration}</p>
        )}
      </div>
    </div>
  );
};

export default FileMetadata;