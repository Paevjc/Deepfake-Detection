import React from 'react';

/**
 * UploadSection Component
 * 
 * Provides a user interface for file upload with preview of selected file name
 * and option to select another file.
 * 
 * @component
 * @param {Object} props - Component props
 * @param {File|null} props.file - Currently selected file or null if no file selected
 * @param {Function} props.onFileChange - Handler function called when file selection changes
 * @param {Function} [props.onReset] - Optional handler function to reset/clear the file selection
 * @returns {JSX.Element} - File upload interface that adapts based on whether a file is selected
 */

const UploadSection = ({ file, onFileChange, onReset }) => {
  return (
    <div className="mb-8">
      <h2 className="text-2xl text-gray-950 font-bold text-center mb-4">Upload Your File</h2>
      <div className="border-2 border-dashed border-gray-300 p-8 text-center rounded-lg bg-white">
        {file ? (
          <div className="text-gray-700 font-medium">
          <p>Chosen file: {file.name}</p>
          <div className="mt-4">
            <label className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md cursor-pointer">
              Choose Another File
              <input 
                type="file" 
                className="hidden"
                onChange={onFileChange}
                accept="video/*,image/*"
              />
            </label>
          </div>
        </div>
        ) : (
          <div>
            <label className="cursor-pointer inline-block">
              <div className="flex flex-col items-center justify-center">
                <svg 
                  className="w-12 h-12 text-gray-400 mb-3" 
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24" 
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    strokeWidth="2" 
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
                <span className="text-gray-600 font-medium">Click to browse files</span>
              </div>
              <input 
                type="file" 
                className="hidden"
                onChange={onFileChange}
                accept="video/*,image/*"
              />
            </label>
          </div>
        )}
        <p className="mt-2 text-sm text-gray-500">
          Select a video or image file to upload
        </p>
      </div>
    </div>
  );
};

export default UploadSection;