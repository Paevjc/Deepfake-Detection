import React, { useState, useEffect } from 'react';

/**
 * FilePreview Component
 *
 * Renders a preview of the uploaded file, supporting both images and videos.
 * Creates object URLs for the file and renders appropriate preview elements.
 *
 * @component
 * @param {Object} props - Component props
 * @param {File|null} props.file - File to preview or null if no file selected
 * @param {boolean} props.detecting - Whether detection is in progress
 * @param {function} props.onDetect - Function to call when detect button is clicked
 * @param {Object|null} props.detectionResults - Results from deepfake detection
 * @param {string|null} props.detectionError - Error message if detection failed
 * @returns {JSX.Element|null} - Returns the file preview or null if no file
 */
const FilePreview = ({ 
  file, 
  detecting = false,
  onDetect = null, 
  detectionResults = null, 
  detectionError = null 
}) => {
  const [previewUrl, setPreviewUrl] = useState(null);
  
  useEffect(() => {
    // Clean up previous URL when file changes
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    
    if (!file) return;
    
    // If file has a url property, use it, otherwise create one
    if (file.url) {
      setPreviewUrl(file.url);
    } else if (file.blob) {
      // If we have a blob directly, create URL from it
      const url = URL.createObjectURL(file.blob);
      setPreviewUrl(url);
      console.log("Created new blob URL from blob:", url);
    } else {
      // For regular File objects
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      console.log("Created new blob URL from file:", url);
    }
    
    // Clean up function to revoke URL when component unmounts or file changes
    return () => {
      if (previewUrl && !file.url) {
        console.log("Revoking URL:", previewUrl);
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [file]);
  
  if (!file || !previewUrl) return null;
  
  const isVideo = file.type.startsWith('video/');
  const isImage = file.type.startsWith('image/');
  
  return (
    <div className="mt-6">
      <h3 className="text-xl text-gray-950 font-semibold text-center mb-4">
        {isVideo ? 'Video Preview' : isImage ? 'Image Preview' : 'File Preview'}
      </h3>
      
      {isVideo && (
        <div className="rounded-lg overflow-hidden max-w-full mx-auto">
          <div className="aspect-video bg-black w-full max-w-3xl mx-auto">
            <video 
              src={previewUrl} 
              controls 
              className="w-full h-full object-contain" 
              preload="metadata"
              onError={(e) => console.error("Video error:", e)}
            />
          </div>
        </div>
      )}
      
      {isImage && (
        <div className="rounded-lg overflow-hidden bg-gray-100 p-2 max-w-md mx-auto">
          <img 
            src={previewUrl} 
            alt="Preview" 
            className="w-full object-contain max-h-96 mx-auto" 
            onError={(e) => console.error("Image error:", e)}
          />
        </div>
      )}
      
      {/* Deepfake detection UI - only shown for videos */}
      {isVideo && onDetect && (
        <div className="mt-4">
          <button
            onClick={onDetect}
            disabled={detecting}
            className="w-full bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded-md transition disabled:opacity-50"
          >
            {detecting ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Analyzing Video...
              </span>
            ) : (
              'Detect Deepfake'
            )}
          </button>
          
          {/* Show detection error if any */}
          {detectionError && (
            <div className="mt-2 text-red-600 bg-red-50 p-3 rounded-md">
              Error: {detectionError}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FilePreview;