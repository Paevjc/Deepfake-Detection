import { useState } from 'react'
import './App.css'
import NavBar from "./components/NavBar"
import FileUploader from "./components/FileUploader"

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
  const [fileUploaded, setFileUploaded] = useState(false)
  const [resetTrigger, setResetTrigger] = useState(0)

  // Function to handle file status changes
  const handleFileStatusChange = (hasFile) => {
    setFileUploaded(hasFile)
  }

  // Function to reset the application state
  const handleReset = () => {
    setFileUploaded(false)
    // Increment resetTrigger to force FileUploader to reset
    setResetTrigger(prev => prev + 1)
  }

  return (
    <>
      <NavBar onReset={handleReset} />
      
      {/* Added tagline */}
      {!fileUploaded && (
        <div className="text-center my-6 px-4">
          <h1 className="text-xl md:text-2xl font-semibold text-gray-800">
            See beyond the surface, uncover digital truth with Deepfake Detector!
          </h1>
        </div>
      )}
      
      {/* Main functionality of file uploading */}
      <div className="container mx-auto my-8 px-4">
        <FileUploader onFileStatusChange={handleFileStatusChange} 
        resetTrigger={resetTrigger}
        />
      </div>
    </>
  )
}

export default App