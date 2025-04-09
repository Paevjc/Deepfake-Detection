import { useState, useEffect } from 'react'
import UploadSection from './UploadSection'
import FileMetadata from './FileMetadata'
import FilePreview from './FilePreview'
import { formatFileSize, formatDuration } from '../utils/fileUtils'
import { createFFmpeg, fetchFile } from '@ffmpeg/ffmpeg'

// Add a new component for showing deepfake detection results
import DeepfakeResults from './DeepfakeResults' // We'll create this component next

let probeOutputString = "";
// Capture logs manually
const ffmpeg = createFFmpeg({ log: false });
// Set up logger to capture 'fferr' (stderr) and 'ffout' (stdout) lines
ffmpeg.setLogger(({ type, message }) => {
  if (type === 'ffout' || type === 'fferr') {
    probeOutputString += message + "\n";
  }
});

/**
 * FileUploader Component
 *
 * Main container component that manages file upload state and coordinates
 * between UploadSection, FileMetadata, and FilePreview components.
 *
 * @component
 * @returns {JSX.Element} - Complete file upload interface with preview and metadata
 */

export default function FileUploader({ onFileStatusChange, resetTrigger = 0 }) {
  const [file, setFile] = useState(null)
  const [fileMetadata, setFileMetadata] = useState(null)
  const [convertedFile, setConvertedFile] = useState(null)
  const [loading, setLoading] = useState(false)
  
  // Add state for deepfake detection
  const [detecting, setDetecting] = useState(false)
  const [detectionResults, setDetectionResults] = useState(null)
  const [detectionError, setDetectionError] = useState(null)

  // Reset state when resetTrigger changes
  useEffect(() => {
    if (resetTrigger > 0) {
      // Reset all state variables
      setFile(null)
      setFileMetadata(null)
      setConvertedFile(null)
      setLoading(false)
      // Also reset deepfake detection states
      setDetecting(false)
      setDetectionResults(null)
      setDetectionError(null)
    }
  }, [resetTrigger])

  // Notify parent component when file status changes
  useEffect(() => {
    if (onFileStatusChange) {
      onFileStatusChange(file !== null)
    }
  }, [file, onFileStatusChange])

  /**
   * Extract basic metadata for images or general file info
   */
  function extractBasicMetadata(selectedFile) {
    // Basic info
    setFileMetadata({
      name: selectedFile.name,
      type: selectedFile.type,
      size: formatFileSize(selectedFile.size),
      lastModified: new Date(selectedFile.lastModified).toLocaleString(),
    })
    
    // If it's an image, get dimensions
    if (selectedFile.type.startsWith('image/')) {
      const objectUrl = URL.createObjectURL(selectedFile)
      const img = new Image()
      img.onload = () => {
        setFileMetadata((prev) => ({
          ...prev,
          dimensions: `${img.naturalWidth} × ${img.naturalHeight}`,
        }))
        URL.revokeObjectURL(objectUrl)
      }
      img.onerror = () => {
        console.error('Failed to load image for metadata extraction')
        URL.revokeObjectURL(objectUrl)
      }
      img.src = objectUrl
    }
    // For video, we'll extract dimensions/duration only after it's converted/confirmed playable
  }

  /**
   * Extract resolution + duration from a playable video Blob
   */
  async function extractVideoMetadata(videoBlob) {
    return new Promise((resolve, reject) => {
      const objectUrl = URL.createObjectURL(videoBlob)
      const video = document.createElement('video')
      video.onloadedmetadata = () => {
        resolve({
          dimensions: `${video.videoWidth} × ${video.videoHeight}`,
          duration: formatDuration(video.duration),
        })
        URL.revokeObjectURL(objectUrl)
      }
      video.onerror = () => {
        reject('Failed to load video after conversion')
        URL.revokeObjectURL(objectUrl)
      }
      video.src = objectUrl
    })
  }

  async function convertVideo(originalFile) {
    try {
      console.log("🚀 Starting video conversion...");
      if (!ffmpeg.isLoaded()) {
        console.log("Loading FFmpeg...");
        await ffmpeg.load();
        console.log("✅ FFmpeg Loaded.");
      }
      const extension = originalFile.name.split('.').pop().toLowerCase();
      const inputFileName = `input.${extension}`;
      
      // Write the input file
      ffmpeg.FS('writeFile', inputFileName, await fetchFile(originalFile));
      console.log("FFmpeg FS after writeFile:", ffmpeg.FS('readdir', '/'));
      
      // Clear old logs each time before we run a command
      probeOutputString = "";
      
      // PROBE the file to see if it's h264, etc.
      await ffmpeg.run('-i', inputFileName);
      
      // Now parse the logs we captured
      console.log("FFmpeg probe output:", probeOutputString);
      const isH264 = probeOutputString.includes("Video: h264") || probeOutputString.includes("Video: hevc");
      const isMpeg4 = probeOutputString.includes("Video: mpeg4"); // older MP4V
      
      // Always convert AVI files regardless of codec
      const isAviFormat = extension === 'avi' || originalFile.type === 'video/x-msvideo';
      
      if (!isH264 || isAviFormat || isMpeg4 || probeOutputString.includes("Video:")) {
        console.log("❗ Converting file to ensure compatibility...");
        
        // Reset logs again before we run the actual encode
        probeOutputString = "";
        
        await ffmpeg.run(
          '-i', inputFileName,
          '-c:v', 'libx264',
          '-preset', 'ultrafast',
          '-pix_fmt', 'yuv420p',
          '-movflags', '+faststart',
          '-profile:v', 'baseline',
          '-level', '3.0',
          '-b:v', '800k',
          '-c:a', 'aac', // Add audio transcoding to ensure audio works
          '-bufsize', '1600k',
          'output.mp4'
        );
        
        // Check logs from the conversion process if needed
        console.log("FFmpeg conversion output:", probeOutputString);
        
        if (!ffmpeg.FS('readdir', '/').includes('output.mp4')) {
          console.error("❌ FFmpeg failed to generate output.mp4!");
          return null;
        }
        
        console.log("✅ Conversion successful! Reading converted file...");
        
        // Get the file data
        const data = ffmpeg.FS('readFile', 'output.mp4');
        
        // Create a blob with explicit type and properties
        // Important: use a more specific MIME type with codecs
        const convertedBlob = new Blob([data.buffer], { 
          type: 'video/mp4; codecs="avc1.42E01E, mp4a.40.2"' 
        });
        
        console.log("🎬 Created blob:", convertedBlob.size, "bytes, type:", convertedBlob.type);
        
        // Cleanup
        ffmpeg.FS('unlink', inputFileName);
        ffmpeg.FS('unlink', 'output.mp4');
        
        return convertedBlob;
      } else {
        console.log("✅ File is already H.264/HEVC in web-compatible container. No conversion needed.");
        
        // Even for files that don't need conversion, create a new blob
        // to ensure proper MIME type and encoding
        const data = ffmpeg.FS('readFile', inputFileName);
        
        const blob = new Blob([data.buffer], { 
          type: 'video/mp4; codecs="avc1.42E01E, mp4a.40.2"' 
        });
        
        console.log("📄 Created blob from original:", blob.size, "bytes, type:", blob.type);
        
        ffmpeg.FS('unlink', inputFileName);
        return blob;
      }
    } catch (error) {
      console.error("❌ FFmpeg conversion error:", error);
      return null;
    }
  }

  
  /**
   * Send video to deepfake detection API
   */
  async function detectDeepfake(videoBlob) {
    setDetecting(true);
    setDetectionError(null);
    
    try {
      // Create a file object from the blob
      const file = new File(
        [videoBlob], 
        convertedFile ? convertedFile.name : 'video.mp4', 
        { type: 'video/mp4' }
      );
      
      // Create form data
      const formData = new FormData();
      formData.append('file', file);
      
      // Send to backend API
      const response = await fetch('/api/analyze-media', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze video');
      }
      
      // Process the response
      const results = await response.json();
      setDetectionResults(results);
      
      // Add detection results to metadata
      setFileMetadata(prev => ({
        ...prev,
        deepfakeAnalysis: {
          isDeepfake: results.is_deepfake,
          confidence: (results.confidence * 100).toFixed(2) + '%',
          processingTime: results.processing_time.toFixed(2) + 's'
        }
      }));
      
    } catch (error) {
      console.error('Deepfake detection error:', error);
      setDetectionError(error.message);
    } finally {
      setDetecting(false);
    }
  }

  /**
   * Handle new file selection
   */
  // Update the handleFileChange function in your FileUploader component:
  async function handleFileChange(e) {
    if (!e.target.files || e.target.files.length === 0) return;
    
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    
    // Reset detection state for new file
    setDetectionResults(null);
    setDetectionError(null);
    
    // Clear previous converted file to release resources
    if (convertedFile && convertedFile.url) {
      // Don't revoke if it's a URL created elsewhere (like in FilePreview)
      if (!convertedFile._urlHandledExternally) {
        console.log("Revoking previous URL:", convertedFile.url);
        URL.revokeObjectURL(convertedFile.url);
      }
    }
    setConvertedFile(null);
    
    // Basic metadata (and image dimensions if needed)
    extractBasicMetadata(selectedFile);
    
    if (selectedFile.type.startsWith('video/')) {
      setLoading(true);
      try {
        const convertedBlob = await convertVideo(selectedFile);
        if (convertedBlob) {
          // Now that it is converted to H.264, read resolution + duration
          try {
            const vidMeta = await extractVideoMetadata(convertedBlob);
            setFileMetadata((prev) => ({ ...prev, ...vidMeta }));
          } catch (err) {
            console.error("Failed to extract video metadata:", err);
          }
          
          // Store the blob directly without creating a URL here
          // Let the FilePreview component handle URL creation
          setConvertedFile({
            name: selectedFile.name.split('.')[0] + '.mp4',
            type: 'video/mp4',
            blob: convertedBlob,
            _urlHandledExternally: true // Flag to indicate URL is handled elsewhere
          });
          
          console.log("✅ Converted file stored in state, size:", convertedBlob.size, "bytes");
        } else {
          console.error("Video conversion failed");
          alert("Sorry, we couldn't process this video file. Please try a different format.");
          setFile(null);
        }
      } catch (error) {
        console.error("Video processing error:", error);
        alert("Error processing video. Please try again with a different file.");
        setFile(null);
      } finally {
        setLoading(false);
      }
    } else {
      // Not a video
      setConvertedFile(null);
    }
  }

  // Add this useEffect for cleanup when component unmounts
  useEffect(() => {
    // Cleanup function to prevent memory leaks
    return () => {
      if (convertedFile && convertedFile.url && !convertedFile._urlHandledExternally) {
        console.log("Component unmounting, revoking URL:", convertedFile.url);
        URL.revokeObjectURL(convertedFile.url);
      }
    };
  }, [convertedFile]);

  // Handle deepfake detection request
  const handleDetectDeepfake = async () => {
    if (!convertedFile || !convertedFile.blob) {
      setDetectionError("No video file available for analysis");
      return;
    }
    
    await detectDeepfake(convertedFile.blob);
  };


  // No file uploaded, stay as single column if not will be a two column layout.
  if (!file) {
    // Single column layout when no file will be centered
    return (
      <div className="flex justify-center items-center">
        <div className="max-w-xl w-full">
          <UploadSection
            file={file}
            onFileChange={handleFileChange}
          />
        </div>
      </div>
    )
  }

  // If there is a file, use two-column layout
  // No file uploaded, stay as single column if not will be a two column layout.
  if (!file) {
    // Single column layout when no file will be centered
    return (
      <div className="flex justify-center items-center">
        <div className="max-w-xl w-full">
          <UploadSection
            file={file}
            onFileChange={handleFileChange}
          />
        </div>
      </div>
    );
  }

  // If there is a file, use two-column layout
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
      {/* Left column for upload + preview */}
      <div className="space-y-6">
        <UploadSection
          file={file}
          onFileChange={handleFileChange}
        />
        
        {loading ? (
          <div className="text-center p-6 bg-gray-100 rounded-lg">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-700 mx-auto mb-4"></div>
            <p className="text-gray-700 font-medium">Converting video... Please wait.</p>
          </div>
        ) : (
          <>
            <FilePreview
              file={convertedFile || file}
              detecting={detecting}
              onDetect={convertedFile && convertedFile.type.startsWith('video/') ? handleDetectDeepfake : null}
              detectionError={detectionError}
            />
            
            {/* Display deepfake results if available */}
            {detectionResults && (
              <DeepfakeResults results={detectionResults} />
            )}
          </>
        )}
      </div>
      
      {/* Right column for metadata */}
      <div>
        {fileMetadata && <FileMetadata metadata={fileMetadata} />}
      </div>
    </div>
  );
}