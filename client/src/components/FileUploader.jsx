import { useState } from 'react';
import UploadSection from './UploadSection';
import FileMetadata from './FileMetadata';
import FilePreview from './FilePreview';
import { formatFileSize } from '../utils/fileUtils';

/**
 * FileUploader Component
 * 
 * Main container component that manages file upload state and coordinates
 * between UploadSection, FileMetadata, and FilePreview components.
 * 
 * @component
 * @returns {JSX.Element} - Complete file upload interface with preview and metadata
 */

export default function FileUploader() {
  const [file, setFile] = useState(null);
  const [fileMetadata, setFileMetadata] = useState(null);

  /**
  * Handles file selection change events
  * 
  * @param {Event} e - File input change event
  * @returns {void}
  */

  function handleFileChange(e) {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      
      // Extract and set metadata
      setFileMetadata({
        name: selectedFile.name,
        type: selectedFile.type,
        size: formatFileSize(selectedFile.size),
        lastModified: new Date(selectedFile.lastModified).toLocaleString()
      });
    }
  }

  return (
    <div className="max-w-2xl mx-auto my-8 bg-gray-50 rounded-lg shadow-lg p-6">
      <UploadSection 
        file={file} 
        onFileChange={handleFileChange} 
      />
      
      {fileMetadata && (
        <>
          <FileMetadata metadata={fileMetadata} />
          <FilePreview file={file} />
        </>
      )}
    </div>
  );
}