import DD_Logo from '../assets/DD_Logo.png';

/**
 * Navigation Bar Component
 * 
 * Provides a simplified top navigation interface with a clickable logo
 * that resets the application state.
 * 
 * @component
 * @param {Object} props - Component props
 * @param {Function} props.onReset - Function to call when logo is clicked
 * @returns {JSX.Element} - Navigation bar UI with centered logo
 */

export default function NavBar({ onReset }) {
  return (
    <nav className="bg-white w-full">
      <div className="mx-auto max-w-7xl px-2 sm:px-6 lg:px-8">
        <div className="flex h-16 md:h-20 items-center justify-center">
          <div 
            className="flex items-center cursor-pointer" 
            onClick={onReset}
            title="Return to home"
          >
            <img
              alt="Deepfake Detector"
              src={DD_Logo}
              className="h-10 sm:h-12 md:h-16 w-auto" // Responsive sizing across breakpoints
            />
          </div>
        </div>
      </div>
    </nav>
  )
}