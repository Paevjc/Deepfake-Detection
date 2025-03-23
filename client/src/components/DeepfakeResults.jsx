import React, { useState } from 'react';

/**
 * DeepfakeResults Component
 * 
 * Displays the results of deepfake detection analysis
 * styled to match your existing UI
 * 
 * @component
 * @param {Object} props
 * @param {Object} props.results - The detection results from the API
 * @returns {JSX.Element}
 */
const DeepfakeResults = ({ results }) => {
  const [showDetails, setShowDetails] = useState(false);
  
  if (!results) return null;
  
  const { 
    is_deepfake, 
    confidence, 
    processing_time, 
    frames_analyzed 
  } = results;
  
  // Format confidence as percentage
  const confidencePercent = (confidence * 100).toFixed(2);
  
  // Determine border color based on result
  const getBorderColor = () => {
    if (is_deepfake) {
      if (confidence > 0.8) return 'border-red-500';
      if (confidence > 0.6) return 'border-orange-500';
      return 'border-yellow-500';
    }
    return 'border-green-500';
  };
  
  return (
    <div className="mt-6">
      <h3 className="text-xl text-gray-950 font-semibold text-center mb-4">
        Detection Results
      </h3>
      <div className={`bg-white text-black p-6 rounded-lg border-2 ${getBorderColor()}`}>
        <p className="font-mono text-lg text-center font-bold">
          {is_deepfake 
            ? '⚠️ Possible Deepfake Detected' 
            : '✓ No Deepfake Detected'}
        </p>
        
        <div className="mt-4">
          <div className="flex justify-between items-center mb-2">
            <span className="font-mono">Confidence:</span>
            <span className="font-mono font-bold">{confidencePercent}%</span>
          </div>
          
          <div className="w-full bg-gray-200 rounded-full h-4">
            <div 
              className={`h-4 rounded-full ${
                is_deepfake 
                  ? confidence > 0.8 ? 'bg-red-600' : confidence > 0.6 ? 'bg-orange-500' : 'bg-yellow-500'
                  : 'bg-green-500'
              }`}
              style={{ width: `${confidencePercent}%` }}
            ></div>
          </div>
        </div>
        
        <div className="mt-6 flex justify-center">
          <button 
            onClick={() => setShowDetails(!showDetails)}
            className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-md transition"
          >
            {showDetails ? 'Hide Details' : 'Show Details'}
          </button>
        </div>
        
        {showDetails && (
          <div className="mt-4 font-mono">
            <p className="mt-2">Processing Time: {processing_time.toFixed(2)} seconds</p>
            <p className="mt-2">Frames Analyzed: {frames_analyzed}</p>
            
            {is_deepfake && (
              <div className="mt-4 p-4 bg-gray-100 rounded-md">
                <p className="font-semibold">What does this mean?</p>
                <p className="mt-2">
                  {confidence > 0.8 
                    ? 'High probability that this video has been manipulated. The AI has detected strong indicators of deepfake technology.'
                    : confidence > 0.6
                      ? 'Moderate indicators of potential manipulation were found in this video.'
                      : 'Some minor signs of manipulation detected, but with lower confidence.'
                  }
                </p>
                <p className="mt-2">
                  Note: This is an automated analysis and should be used as one of multiple verification methods.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default DeepfakeResults;