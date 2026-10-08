
import { useEffect, useRef, useState } from "react";
import "./App.css";

function App() {
  // Selected satellite images
  const [imageT1, setImageT1] = useState(null);
  const [imageT2, setImageT2] = useState(null);

  // Image preview URLs
  const [previewT1, setPreviewT1] = useState(null);
  const [previewT2, setPreviewT2] = useState(null);

  // File input references
  const inputT1Ref = useRef(null);
  const inputT2Ref = useRef(null);

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");

  // Generate preview for T1
  useEffect(() => {
    if (!imageT1) {
      setPreviewT1(null);
      return;
    }

    const url = URL.createObjectURL(imageT1);
    setPreviewT1(url);

    return () => URL.revokeObjectURL(url);
  }, [imageT1]);

  // Generate preview for T2
  useEffect(() => {
    if (!imageT2) {
      setPreviewT2(null);
      return;
    }

    const url = URL.createObjectURL(imageT2);
    setPreviewT2(url);

    return () => URL.revokeObjectURL(url);
  }, [imageT2]);

  // Remove T1 image
  const removeImageT1 = () => {
    setImageT1(null);

    if (inputT1Ref.current) {
      inputT1Ref.current.value = "";
    }
  };

  // Remove T2 image
  const removeImageT2 = () => {
    setImageT2(null);

    if (inputT2Ref.current) {
      inputT2Ref.current.value = "";
    }
  };


const handleAnalyze = async () => {
  if (!imageT1 || !imageT2) return;

  setLoading(true);
  setError("");
  setResults(null);

  try {
    const formData = new FormData();

    formData.append("image_t1", imageT1);
    formData.append("image_t2", imageT2);

    const response = await fetch(
      "http://127.0.0.1:8000/analyze",
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      let message = `Analysis failed (${response.status})`;

      try {
        const errorData = await response.json();
        if (typeof errorData.detail === "string") {
          message = errorData.detail;
        }
      } catch {
        // Use the default error message.
      }

      throw new Error(message);
    }

    const data = await response.json();
    setResults(data);
  } catch (err) {
    setError(err.message || "Unable to analyze images.");
  } finally {
    setLoading(false);
  }
};


  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1>TerraVision</h1>
        <p>AI-Powered Land Change Detection</p>
      </header>

      <main className="container">
        <section className="upload-section">
          <h2>Analyze Land Change</h2>

          <p>
            Upload satellite images of the same region
            captured at two different time periods.
          </p>

          <div className="upload-grid">
            {/* Earlier Image */}
            <div className="upload-card">
              <h3>Earlier Image (T1)</h3>

              <input
                ref={inputT1Ref}
                type="file"
                accept="image/png,image/jpeg"
                onChange={(e) =>
                  setImageT1(e.target.files?.[0] || null)
                }
              />

              {previewT1 && (
                <div className="preview-container">
                  <img
                    src={previewT1}
                    alt="Earlier satellite image"
                    className="image-preview"
                  />

                  <button
                    type="button"
                    className="remove-button"
                    onClick={removeImageT1}
                  >
                    Remove Image
                  </button>
                </div>
              )}
            </div>

            {/* Later Image */}
            <div className="upload-card">
              <h3>Later Image (T2)</h3>

              <input
                ref={inputT2Ref}
                type="file"
                accept="image/png,image/jpeg"
                onChange={(e) =>
                  setImageT2(e.target.files?.[0] || null)
                }
              />

              {previewT2 && (
                <div className="preview-container">
                  <img
                    src={previewT2}
                    alt="Later satellite image"
                    className="image-preview"
                  />

                  <button
                    type="button"
                    className="remove-button"
                    onClick={removeImageT2}
                  >
                    Remove Image
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Analyze Button */}
          <button
            type="button"
            className="analyze-button"
            onClick={handleAnalyze}
            disabled={!imageT1 || !imageT2 || loading}
          >
            {loading ? "Analyzing Images..." : "Analyze Changes"}

          </button>
          {error && (
            <p style={{ color: "#dc2626" }} role="alert">
              {error}
            </p>
          )}

          {results && (
            <div className="results-section">
              <h2>Analysis Results</h2>

              <p>
                <strong>Detected Change:</strong>{" "}
                {results.change_percentage.toFixed(2)}%
              </p>

              <p>
                <strong>Changed Pixels:</strong>{" "}
                {results.changed_pixels.toLocaleString()}
              </p>

              <p>
                <strong>Total Pixels:</strong>{" "}
                {results.total_pixels.toLocaleString()}
              </p>


              <h3>AI-Generated Land-Cover Maps</h3>

              <div className="results-grid">
                <div className="result-card">
                  <h4>Land Cover — T1</h4>
                  <img
                    src={results.visualizations.segmentation_t1}
                    alt="Land-cover segmentation at T1"
                  />
                </div>

                <div className="result-card">
                  <h4>Land Cover — T2</h4>
                  <img
                    src={results.visualizations.segmentation_t2}
                    alt="Land-cover segmentation at T2"
                  />
                </div>

                <div className="result-card">
                  <h4>Detected Changes</h4>
                  <img
                    src={results.visualizations.change_map}
                    alt="Predicted land-cover changes"
                  />
                </div>
              </div>

              <h3>Land-Cover Statistics</h3>

              <div className="table-wrapper">
                <table className="stats-table">
                  <thead>
                    <tr>
                      <th>Land-Cover Class</th>
                      <th>Before (T1)</th>
                      <th>After (T2)</th>
                      <th>Net Change</th>
                    </tr>
                  </thead>

                  <tbody>
                    {Object.entries(results.land_cover).map(
                      ([className, stats]) => (
                        <tr key={className}>
                          <td>{className}</td>
                          <td>{stats.t1_percentage.toFixed(2)}%</td>
                          <td>{stats.t2_percentage.toFixed(2)}%</td>
                          <td>
                            {stats.net_change > 0 ? "+" : ""}
                            {stats.net_change.toFixed(2)} pp
                          </td>
                        </tr>
                      )
                    )}
                  </tbody>
                </table>
              </div>

              <h3>Major Land-Cover Transitions</h3>

              <div className="transitions-list">
                {Object.entries(results.transitions)
                  .slice(0, 5)
                  .map(([transition, stats]) => (
                    <div className="transition-item" key={transition}>
                      <span>{transition.replace(" -> ", " → ")}</span>
                      <strong>{stats.percentage.toFixed(2)}%</strong>
                    </div>
                  ))}
              </div>

            </div>
          )}
          <p className="upload-note">
            For accurate change detection, both images
            must show the same geographical area and
            be properly aligned.
          </p>
        </section>
      </main>
    </div>
  );
}

export default App;
