import { useState } from 'react'
import './App.css'
import NavBar from "./components/NavBar";
import UploadSection from "./components/UploadSection";
import FileMetadata from "./components/FileMetadata";
import FilePreview from "./components/FilePreview";
import { formatFileSize, formatDuration } from './utils/fileUtils';

/**
 * Main App Component
 * 
 * Root component that manages application state and layout.
 * Handles file uploads, extracts metadata, and coordinates the UI components.
 * 
 * @component
 * @returns {JSX.Element} - Complete application UI
 */

function App() {
  const [file, setFile] = useState(null);
  const [fileMetadata, setFileMetadata] = useState(null);

  /**
 * Handles file selection change events and extracts metadata
 * 
 * Processes the selected file to extract basic metadata immediately,
 * then asynchronously extracts advanced metadata (like dimensions and duration)
 * for images and videos.
 * 
 * @param {Event} e - File input change event
 * @returns {void}
 */

  function handleFileChange(e) {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      
      // Basic metadata that doesn't require async operations
      const basicMetadata = {
        name: selectedFile.name,
        type: selectedFile.type,
        size: formatFileSize(selectedFile.size),
        lastModified: new Date(selectedFile.lastModified).toLocaleString()
      };
      
      // Set the basic metadata immediately
      setFileMetadata(basicMetadata);
      
      // Then try to get additional metadata based on file type
      try {
        if (selectedFile.type.startsWith('image/')) {
          // For images - get dimensions safely
          const objectUrl = URL.createObjectURL(selectedFile);
          const img = new Image();
          
          img.onload = () => {
            try {
              setFileMetadata(prev => ({
                ...prev,
                dimensions: `${img.naturalWidth} × ${img.naturalHeight}`
              }));
            } catch (error) {
              console.error("Error updating image metadata:", error);
            } finally {
              URL.revokeObjectURL(objectUrl);
            }
          };
          
          img.onerror = () => {
            console.error("Failed to load image for metadata extraction");
            URL.revokeObjectURL(objectUrl);
          };
          
          img.src = objectUrl;
        } 
        else if (selectedFile.type.startsWith('video/')) {
          // For videos - get dimensions and duration safely
          const objectUrl = URL.createObjectURL(selectedFile);
          const video = document.createElement('video');
          
          video.onloadedmetadata = () => {
            try {
              setFileMetadata(prev => ({
                ...prev,
                dimensions: `${video.videoWidth} × ${video.videoHeight}`,
                duration: formatDuration(video.duration)
              }));
            } catch (error) {
              console.error("Error updating video metadata:", error);
            } finally {
              URL.revokeObjectURL(objectUrl);
            }
          };
          
          video.onerror = () => {
            console.error("Failed to load video for metadata extraction");
            URL.revokeObjectURL(objectUrl);
          };
          
          video.src = objectUrl;
        }
      } catch (error) {
        console.error("Error during metadata extraction:", error);
        // We already set basic metadata, so the app won't break
      }
    }
  }

  return (
    <>
      <NavBar />
      <div className="container mx-auto my-8 px-4">
        {!file ? (
          // Initial centered upload state
          <div className="max-w-xl mx-auto">
            <UploadSection 
              file={file} 
              onFileChange={handleFileChange} 
            />
          </div>
        ) : (
          // Side-by-side layout after file is uploaded
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Left column for upload and preview */}
            <div className="space-y-6">
              <UploadSection 
                file={file} 
                onFileChange={handleFileChange} 
              />
              <FilePreview file={file} />
            </div>
            
            {/* Right column for metadata */}
            <div>
              {fileMetadata && <FileMetadata metadata={fileMetadata} />}
            </div>
          </div>
        )}
      </div>
    </>
  )
}

export default App