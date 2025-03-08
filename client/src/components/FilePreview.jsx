import React from 'react';

/**
 * FilePreview Component
 * 
 * Renders a preview of the uploaded file, supporting both images and videos.
 * Creates object URLs for the file and renders appropriate preview elements.
 * 
 * @component
 * @param {Object} props - Component props
 * @param {File|null} props.file - File to preview or null if no file selected
 * @returns {JSX.Element|null} - Returns the file preview or null if no file
 */

const FilePreview = ({ file }) => {
  if (!file) return null;

  const isVideo = file.type.startsWith('video/');
  const isImage = file.type.startsWith('image/');
  const fileUrl = URL.createObjectURL(file);

  return (
    <div className="mt-6">
      <h3 className="text-xl text-gray-950 font-semibold text-center mb-4">
        {isVideo ? 'Video Preview' : isImage ? 'Image Preview' : 'File Preview'}
      </h3>
      
      {isVideo && (
        <div className="rounded-lg overflow-hidden max-w-full mx-auto">
          {/* Container for video */}
          <div className="aspect-video bg-black w-full max-w-3xl mx-auto">
            <video 
              src={fileUrl} 
              controls 
              className="w-full h-full object-contain"
            />
          </div>
        </div>
      )}
      
      {isImage && (
        <div className="rounded-lg overflow-hidden bg-gray-100 p-2 max-w-md mx-auto">
          <img 
            src={fileUrl} 
            alt="Preview" 
            className="w-full object-contain max-h-96 mx-auto"
          />
        </div>
      )}
    </div>
  );
};

export default FilePreview;