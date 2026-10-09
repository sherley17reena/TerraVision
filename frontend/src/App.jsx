
import { useEffect, useRef, useState } from "react";
import "./App.css";

import BeforeAfterSlider from "./BeforeAfterSlider";
import BuildingChangeOverlay from "./BuildingChangeOverlay";

import {
  downloadReport,
  downloadMask,
  downloadOverlay,
} from "./downloadResults";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [analysisMode, setAnalysisMode] = useState("land-cover");

  const [imageT1, setImageT1] = useState(null);
  const [imageT2, setImageT2] = useState(null);

  const [previewT1, setPreviewT1] = useState(null);
  const [previewT2, setPreviewT2] = useState(null);

  const inputT1Ref = useRef(null);
  const inputT2Ref = useRef(null);

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");

  // Generate image previews and release object URLs.
  useEffect(() => {
    if (!imageT1) {
      setPreviewT1(null);
      return;
    }

    const url = URL.createObjectURL(imageT1);
    setPreviewT1(url);

    return () => URL.revokeObjectURL(url);
  }, [imageT1]);

  useEffect(() => {
    if (!imageT2) {
      setPreviewT2(null);
      return;
    }

    const url = URL.createObjectURL(imageT2);
    setPreviewT2(url);

    return () => URL.revokeObjectURL(url);
  }, [imageT2]);

  const resetResults = () => {
    setResults(null);
    setError("");
  };

  const changeMode = (mode) => {
    setAnalysisMode(mode);
    resetResults();
  };

  const removeImageT1 = () => {
    setImageT1(null);
    resetResults();

    if (inputT1Ref.current) {
      inputT1Ref.current.value = "";
    }
  };

  const removeImageT2 = () => {
    setImageT2(null);
    resetResults();

    if (inputT2Ref.current) {
      inputT2Ref.current.value = "";
    }
  };

  // Send images to the selected FastAPI endpoint.
  const handleAnalyze = async () => {
    if (!imageT1 || !imageT2 || loading) return;

    setLoading(true);
    setError("");
    setResults(null);

    const selectedMode = analysisMode;

    const endpoint =
      selectedMode === "buildings"
        ? "/analyze/buildings"
        : "/analyze";

    try {
      const formData = new FormData();

      formData.append("image_t1", imageT1);
      formData.append("image_t2", imageT2);

      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let message = `Analysis failed (${response.status})`;

        try {
          const errorData = await response.json();

          if (typeof errorData.detail === "string") {
            message = errorData.detail;
          }
        } catch {
          // Keep the default error message.
        }

        throw new Error(message);
      }

      const data = await response.json();

      setResults({
        mode: selectedMode,
        data,
      });
    } catch (err) {
      setError(err.message || "Unable to analyze images.");
    } finally {
      setLoading(false);
    }
  };

  const handleOverlayDownload = async () => {
    if (!previewT2 || !results?.data?.change_mask) return;

    try {
      setError("");

      await downloadOverlay(
        previewT2,
        results.data.change_mask
      );
    } catch (err) {
      setError(
        err.message || "Could not download the red overlay."
      );
    }
  };

  const resultData = results?.data;
  const resultMode = results?.mode;
  const isBuildingMode = analysisMode === "buildings";

  return (
    <div className="app">
      {/* HEADER */}
      <header className="header">
        <h1>TerraVision</h1>

        <p>
          AI-Powered Land Change Detection & Environmental Intelligence
        </p>
      </header>

      <main className="container">
        <section className="upload-section">
          <h2>Satellite Change Analysis</h2>

          <p>
            Upload satellite images of the same geographical region
            captured at two different time periods.
          </p>

          {/* ANALYSIS MODE SELECTOR */}
          <div className="analysis-mode-selector">
            <button
              type="button"
              className={
                analysisMode === "land-cover"
                  ? "mode-button active"
                  : "mode-button"
              }
              onClick={() => changeMode("land-cover")}
              disabled={loading}
            >
              Land-Cover Analysis
            </button>

            <button
              type="button"
              className={
                isBuildingMode
                  ? "mode-button active"
                  : "mode-button"
              }
              onClick={() => changeMode("buildings")}
              disabled={loading}
            >
              Building Change Detection
            </button>
          </div>

          <p className="mode-description">
            {isBuildingMode
              ? "Detect building changes using a Temporal U-Net trained on paired satellite images."
              : "Compare seven predicted land-cover categories across two satellite images."}
          </p>

          {/* IMAGE UPLOADS */}
          <div className="upload-grid">
            {/* EARLIER IMAGE */}
            <div className="upload-card">
              <h3>Earlier Image (T1)</h3>

              <input
                ref={inputT1Ref}
                type="file"
                accept="image/png,image/jpeg"
                onChange={(e) => {
                  setImageT1(e.target.files?.[0] || null);
                  resetResults();
                }}
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

            {/* LATER IMAGE */}
            <div className="upload-card">
              <h3>Later Image (T2)</h3>

              <input
                ref={inputT2Ref}
                type="file"
                accept="image/png,image/jpeg"
                onChange={(e) => {
                  setImageT2(e.target.files?.[0] || null);
                  resetResults();
                }}
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

          {/* ANALYZE BUTTON */}
          <button
            type="button"
            className="analyze-button"
            onClick={handleAnalyze}
            disabled={!imageT1 || !imageT2 || loading}
          >
            {loading
              ? "Analyzing Images..."
              : isBuildingMode
                ? "Detect Building Changes"
                : "Analyze Land-Cover Changes"}
          </button>

          {/* ERROR MESSAGE */}
          {error && (
            <p className="analysis-error" role="alert">
              {error}
            </p>
          )}

          {/* ANALYSIS RESULTS */}
          {resultData && (
            <div className="results-section">
              <h2>
                {resultMode === "buildings"
                  ? "Building Change Detection Results"
                  : "Land-Cover Analysis Results"}
              </h2>

              {/* DOWNLOAD BUTTONS */}
              <div className="download-actions">
                <button
                  type="button"
                  className="download-button"
                  onClick={() =>
                    downloadReport(resultData, resultMode)
                  }
                >
                  Download JSON Report
                </button>

                {resultMode === "buildings" && (
                  <>
                    <button
                      type="button"
                      className="download-button"
                      onClick={() =>
                        downloadMask(resultData.change_mask)
                      }
                    >
                      Download Change Mask
                    </button>

                    <button
                      type="button"
                      className="download-button"
                      onClick={handleOverlayDownload}
                    >
                      Download Red Overlay
                    </button>
                  </>
                )}
              </div>

              {/* SUMMARY STATISTICS */}
              <div className="summary-grid">
                <div className="summary-card">
                  <span>Predicted Change</span>

                  <strong>
                    {Number(
                      resultData.change_percentage
                    ).toFixed(2)}
                    %
                  </strong>
                </div>

                <div className="summary-card">
                  <span>Changed Pixels</span>

                  <strong>
                    {resultData.changed_pixels.toLocaleString()}
                  </strong>
                </div>

                <div className="summary-card">
                  <span>Total Pixels</span>

                  <strong>
                    {resultData.total_pixels.toLocaleString()}
                  </strong>
                </div>
              </div>

              {/* BUILDING CHANGE DETECTION */}
              {resultMode === "buildings" ? (
                <>
                  {/* BEFORE/AFTER SLIDER */}
                  <BeforeAfterSlider
                    beforeImage={previewT1}
                    afterImage={previewT2}
                  />

                  {/* CORRECTED RED CHANGE OVERLAY */}
                  <h3>Building Change Overlay</h3>

                  <BuildingChangeOverlay
                    afterImage={previewT2}
                    base64Mask={resultData.change_mask}
                  />

                  <p className="result-explanation">
                    Red highlighted areas indicate predicted
                    building changes between the two satellite
                    images.
                  </p>

                  {/* BUILDING CHANGE MAPS */}
                  <h3>AI-Detected Building Changes</h3>

                  <div className="results-grid">
                    <div className="result-card">
                      <h4>Before (T1)</h4>

                      <img
                        src={previewT1}
                        alt="Satellite image before changes"
                      />
                    </div>

                    <div className="result-card">
                      <h4>After (T2)</h4>

                      <img
                        src={previewT2}
                        alt="Satellite image after changes"
                      />
                    </div>

                    <div className="result-card">
                      <h4>Predicted Building Change Mask</h4>

                      <img
                        src={`data:image/png;base64,${resultData.change_mask}`}
                        alt="Predicted building changes"
                        className="change-mask-image"
                      />
                    </div>
                  </div>

                  <p className="result-explanation">
                    White pixels indicate predicted building
                    changes. Black pixels indicate areas predicted
                    to be unchanged.
                  </p>

                  <p className="upload-note">
                    Model: Temporal U-Net | LEVIR-CD Test F1:
                    83.16% | Test IoU: 71.18%. These scores
                    describe dataset-level evaluation performance,
                    not confidence for this particular upload.
                  </p>
                </>
              ) : (
                <>
                  {/* LAND-COVER SEGMENTATION MAPS */}
                  <h3>AI-Generated Land-Cover Maps</h3>

                  <div className="results-grid">
                    <div className="result-card">
                      <h4>Land Cover — T1</h4>

                      <img
                        src={
                          resultData.visualizations.segmentation_t1
                        }
                        alt="Land-cover segmentation at T1"
                      />
                    </div>

                    <div className="result-card">
                      <h4>Land Cover — T2</h4>

                      <img
                        src={
                          resultData.visualizations.segmentation_t2
                        }
                        alt="Land-cover segmentation at T2"
                      />
                    </div>

                    <div className="result-card">
                      <h4>Detected Changes</h4>

                      <img
                        src={
                          resultData.visualizations.change_map
                        }
                        alt="Predicted land-cover changes"
                      />
                    </div>
                  </div>

                  {/* LAND-COVER STATISTICS */}
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
                        {Object.entries(
                          resultData.land_cover
                        ).map(([className, stats]) => (
                          <tr key={className}>
                            <td>{className}</td>

                            <td>
                              {stats.t1_percentage.toFixed(2)}%
                            </td>

                            <td>
                              {stats.t2_percentage.toFixed(2)}%
                            </td>

                            <td>
                              {stats.net_change > 0 ? "+" : ""}
                              {stats.net_change.toFixed(2)} pp
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* LAND-COVER TRANSITIONS */}
                  <h3>Major Land-Cover Transitions</h3>

                  <div className="transitions-list">
                    {Object.entries(resultData.transitions)
                      .slice(0, 5)
                      .map(([transition, stats]) => (
                        <div
                          className="transition-item"
                          key={transition}
                        >
                          <span>
                            {transition.replace(
                              " -> ",
                              " → "
                            )}
                          </span>

                          <strong>
                            {stats.percentage.toFixed(2)}%
                          </strong>
                        </div>
                      ))}
                  </div>
                </>
              )}
            </div>
          )}

          {/* MODEL LIMITATIONS */}
          <p className="upload-note">
            For meaningful results, both images must show the
            same geographical area, be properly aligned, and
            have comparable spatial resolution. Building-change
            detection is trained specifically on LEVIR-CD imagery.
          </p>
        </section>
      </main>
    </div>
  );
}

export default App;
