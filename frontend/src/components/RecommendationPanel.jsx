import { useState } from "react";
import { getRecommendations, getImageUrl } from "../api";

const TONE_GRADIENTS = {
  Fair:   "linear-gradient(135deg, #fce4ec, #f3e5f5, #e8eaf6)",
  Light:  "linear-gradient(135deg, #fff3e0, #fbe9e7, #fce4ec)",
  Medium: "linear-gradient(135deg, #fff8e1, #fff3e0, #f1f8e9)",
  Tan:    "linear-gradient(135deg, #ffe0b2, #ffcc80, #fff9c4)",
  Deep:   "linear-gradient(135deg, #d7ccc8, #bcaaa4, #efebe9)",
};

export default function RecommendationPanel({ garments }) {
  const [photoFile, setPhotoFile] = useState(null);
  const [photoUrl, setPhotoUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePhoto = (e) => {
    const file = e.target.files[0];
    if (file) {
      setPhotoFile(file);
      setPhotoUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!photoFile) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getRecommendations(photoFile);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const recommendedGarments = result
    ? garments.filter((g) => result.recommended_garment_ids.includes(g.id))
    : [];

  const avoidedGarments = result
    ? garments.filter(
        (g) =>
          result.all_scores[g.id] !== undefined &&
          result.all_scores[g.id] < 0.5 &&
          !result.recommended_garment_ids.includes(g.id)
      )
    : [];

  return (
    <div className="reco-panel">
      {/* ── Upload & Analyze ──────────────────────── */}
      <div className="reco-upload-row">
        <div className="reco-photo-upload glass">
          <input
            type="file"
            accept="image/*"
            onChange={handlePhoto}
            id="reco-photo-input"
            style={{ display: "none" }}
          />
          <label htmlFor="reco-photo-input" className="reco-photo-label">
            {photoUrl ? (
              <img src={photoUrl} alt="Your Photo" className="reco-photo-preview" />
            ) : (
              <div className="reco-photo-placeholder">
                <span className="reco-upload-icon">🤳</span>
                <span>Upload a selfie</span>
                <span className="reco-photo-hint">Clear, well-lit photo works best</span>
              </div>
            )}
          </label>
        </div>

        <div className="reco-action-area">
          <button
            className="btn btn-primary reco-analyze-btn"
            onClick={handleAnalyze}
            disabled={!photoFile || loading}
          >
            {loading ? (
              <>
                <span className="reco-spinner"></span>
                Analyzing skin tone…
              </>
            ) : (
              <>🔬 Analyze My Skin Tone</>
            )}
          </button>
          <p className="reco-action-hint">
            Our AI uses HSV/LAB color science to detect your skin tone and recommend the most flattering garments from your wardrobe.
          </p>
        </div>
      </div>

      {/* ── Error ─────────────────────────────────── */}
      {error && (
        <div className="reco-error glass">
          <span>⚠️</span> {error}
        </div>
      )}

      {/* ── Results ───────────────────────────────── */}
      {result && (
        <div className="reco-results" style={{ animationDelay: "0.1s" }}>
          {/* Skin Tone Card */}
          <div
            className="reco-tone-card glass"
            style={{ background: TONE_GRADIENTS[result.skin_tone] || TONE_GRADIENTS.Medium }}
          >
            <div className="reco-tone-header">
              <span className="reco-tone-emoji">{result.skin_tone_emoji}</span>
              <div>
                <h3 className="reco-tone-title">{result.skin_tone} Skin Tone</h3>
                <p className="reco-tone-desc">{result.description}</p>
              </div>
            </div>
            <div className="reco-tone-tip">
              <span className="reco-tip-icon">💡</span>
              <span>{result.style_tip}</span>
            </div>
            <div className="reco-best-colors">
              <span className="reco-colors-label">Your power colors:</span>
              <div className="reco-color-chips">
                {result.best_colors.map((c) => (
                  <span key={c} className="reco-color-chip">{c}</span>
                ))}
              </div>
            </div>
          </div>

          {/* Recommended Garments */}
          <div className="reco-section">
            <h3 className="reco-section-title">
              <span className="reco-match-icon">✅</span>
              Perfect Matches
              <span className="reco-count">{recommendedGarments.length}</span>
            </h3>
            {recommendedGarments.length > 0 ? (
              <div className="reco-garment-grid">
                {recommendedGarments.map((g) => (
                  <div key={g.id} className="reco-garment-card glass">
                    <div className="reco-garment-score-badge reco-good">
                      {Math.round((result.all_scores[g.id] || 0) * 100)}%
                    </div>
                    <div className="reco-garment-img">
                      {g.image_path ? (
                        <img src={getImageUrl(g.image_path)} alt={g.name} loading="lazy" />
                      ) : (
                        <span className="reco-garment-placeholder">👚</span>
                      )}
                    </div>
                    <div className="reco-garment-info">
                      <span className="reco-garment-name">{g.name}</span>
                      <span className="reco-garment-color">
                        <span
                          className="color-dot"
                          style={{ background: g.dominant_color?.hex || "#888" }}
                        />
                        {g.dominant_color?.name || "—"}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="reco-empty">
                <span>🛍️</span>
                <p>No strong matches yet — try adding garments in your recommended colors!</p>
              </div>
            )}
          </div>

          {/* Garments to Avoid */}
          {avoidedGarments.length > 0 && (
            <div className="reco-section reco-avoid-section">
              <h3 className="reco-section-title">
                <span className="reco-match-icon">⚡</span>
                Less Ideal
                <span className="reco-count reco-count-warn">{avoidedGarments.length}</span>
              </h3>
              <div className="reco-garment-grid reco-garment-grid-muted">
                {avoidedGarments.map((g) => (
                  <div key={g.id} className="reco-garment-card glass reco-garment-muted">
                    <div className="reco-garment-score-badge reco-low">
                      {Math.round((result.all_scores[g.id] || 0) * 100)}%
                    </div>
                    <div className="reco-garment-img">
                      {g.image_path ? (
                        <img src={getImageUrl(g.image_path)} alt={g.name} loading="lazy" />
                      ) : (
                        <span className="reco-garment-placeholder">👚</span>
                      )}
                    </div>
                    <div className="reco-garment-info">
                      <span className="reco-garment-name">{g.name}</span>
                      <span className="reco-garment-color">
                        <span
                          className="color-dot"
                          style={{ background: g.dominant_color?.hex || "#888" }}
                        />
                        {g.dominant_color?.name || "—"}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
