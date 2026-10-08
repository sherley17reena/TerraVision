
import { useState } from "react";
import "./BeforeAfterSlider.css";

function BeforeAfterSlider({ beforeImage, afterImage }) {
  const [position, setPosition] = useState(50);

  return (
    <div className="comparison-section">
      <h3>Before & After Comparison</h3>

      <p className="comparison-description">
        Drag the slider to compare satellite images
        before and after the detected changes.
      </p>

      <div className="comparison-container">
        <img
          src={afterImage}
          alt="Satellite image after changes"
          className="comparison-image"
        />

        <div
          className="comparison-before"
          style={{
            clipPath: `inset(0 ${100 - position}% 0 0)`,
          }}
        >
          <img
            src={beforeImage}
            alt="Satellite image before changes"
            className="comparison-image"
          />
        </div>

        <div
          className="comparison-divider"
          style={{ left: `${position}%` }}
          aria-hidden="true"
        >
          <div className="comparison-handle">↔</div>
        </div>

        <span className="comparison-label comparison-label-before">
          Before
        </span>

        <span className="comparison-label comparison-label-after">
          After
        </span>

        <input
          type="range"
          min="0"
          max="100"
          value={position}
          onChange={(e) => setPosition(Number(e.target.value))}
          className="comparison-range"
          aria-label="Adjust before and after satellite image comparison"
        />
      </div>
    </div>
  );
}

export default BeforeAfterSlider;
