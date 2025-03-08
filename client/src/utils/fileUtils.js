/**
 * Utility functions for file operations
 * 
 * Contains helper functions for formatting file metadata.
 */

/**
 * Formats file size in bytes to a human-readable format
 * @param {number} sizeInBytes - The file size in bytes
 * @returns {string} Formatted file size (e.g., "2.5 MB")
 */

export function formatFileSize(sizeInBytes) {
    if (sizeInBytes < 1024) {
      return sizeInBytes + ' bytes';
    } else if (sizeInBytes < 1024 * 1024) {
      return (sizeInBytes / 1024).toFixed(2) + ' KB';
    } else if (sizeInBytes < 1024 * 1024 * 1024) {
      return (sizeInBytes / (1024 * 1024)).toFixed(2) + ' MB';
    } else {
      return (sizeInBytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
    }
  }
  
  /**
   * Formats duration in seconds to a human-readable format
   * @param {number} seconds - The duration in seconds
   * @returns {string} Formatted duration
   */
  export function formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  }