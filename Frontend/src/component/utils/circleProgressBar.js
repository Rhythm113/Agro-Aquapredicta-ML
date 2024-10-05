import React, { useEffect, useRef, useState } from 'react';
import './circleProgressBar.css';

const CircleProgressBar = ({ content }) => {
  const { avgPercentage, title } = content;
  const [progress, setProgress] = useState(0); // State for animating progress
  const circleRef = useRef(null); // Reference for each circle

  useEffect(() => {
    if (circleRef.current) {
      // Get the width of the circle element
      const width = circleRef.current.offsetWidth;
      // Set the height to match the width to keep the circle aspect ratio
      circleRef.current.style.height = `${width}px`;
    }
    
    let degree = 0;
    const targetDegree = parseInt(avgPercentage);
    const color = targetDegree >= 70 ? "#3FA62A" : targetDegree >= 30 ? "#F8B812" : "#E00219";

    const interval = setInterval(() => {
      degree += 1;
      if (degree > targetDegree) {
        clearInterval(interval);
        return;
      }
      setProgress(degree); // Update the progress state

      // Access and modify the circle's background using ref
      if (circleRef.current) {
        circleRef.current.style.background = `conic-gradient(${color} ${degree}%, #222 0%)`;
      }
    }, 50);

    return () => clearInterval(interval); // Clean up the interval on component unmount
  }, [avgPercentage]);

  return (
    <div className='circleProgressBar'>
      <div ref={circleRef} className='circle'> {/* Attach ref here */}
        <h2 className='number'>{progress}%</h2> {/* Use the animated progress */}
      </div>
    </div>
  );
};

export default CircleProgressBar;
