import { useState, useEffect, useCallback } from "react";
import UploadZone from "./components/UploadZone";
import WardrobeGrid from "./components/WardrobeGrid";
import { fetchGarments, deleteGarment, tryOnGarment } from "./api";
import "./index.css";

const CATEGORIES = [
  { value: "all", label: "All" },
  { value: "top", label: "Tops" },
  { value: "bottom", label: "Bottoms" },
  { value: "dress", label: "Dresses" },
  { value: "shoes", label: "Shoes" },
  { value: "accessory", label: "Accessories" },
  { value: "other", label: "Other" },
];

export default function App() {
  const [garments, setGarments] = useState([]);
  const [filter, setFilter] = useState("all");
  const [toast, setToast] = useState(null);

  // Try-On State
  const [userPhotoFile, setUserPhotoFile] = useState(null);
  const [userPhotoUrl, setUserPhotoUrl] = useState(null);
  const [tryOnResultUrl, setTryOnResultUrl] = useState(null);
  const [isTryingOn, setIsTryingOn] = useState(false);
  const [styleFeedback, setStyleFeedback] = useState(null);

  const showToast = (msg, type = "success") => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3500);
  };

  const loadGarments = useCallback(async () => {
    try {
      const data = await fetchGarments();
      setGarments(data);
    } catch {
      /* backend might not be running yet */
    }
  }, []);

  useEffect(() => { loadGarments(); }, [loadGarments]);

  const handleUploadComplete = (result) => {
    showToast(`"${result.name}" added — detected ${result.dominant_color?.name} 🎨`);
    loadGarments();
  };

  const handleDelete = async (id) => {
    try {
      await deleteGarment(id);
      setGarments((prev) => prev.filter((g) => g.id !== id));
      showToast("Garment removed", "success");
    } catch {
      showToast("Failed to delete", "error");
    }
  };

  const handleUserPhotoUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUserPhotoFile(file);
      setUserPhotoUrl(URL.createObjectURL(file));
      setTryOnResultUrl(null);
      setStyleFeedback(null);
      showToast("User photo uploaded successfully");
    }
  };

  const handleTryOn = async (garmentId) => {
    if (!userPhotoFile) {
      showToast("Please upload a Base Photo first to try on clothes!", "error");
      return;
    }

    setIsTryingOn(true);
    setTryOnResultUrl(null);
    setStyleFeedback(null);
    showToast("🧠 AI model is generating your try-on — this takes ~60 seconds...", "info");

    try {
      const data = await tryOnGarment(garmentId, userPhotoFile);
      setTryOnResultUrl(data.tryon_image_base64);
      setStyleFeedback(data.style_feedback);
      showToast("✅ AI Try-On complete!", "success");
    } catch (err) {
      showToast(err.message || "Failed to process Try-On", "error");
    } finally {
      setIsTryingOn(false);
    }
  };

  const colorCounts = {};
  garments.forEach((g) => {
    const c = g.dominant_color?.name || "Unknown";
    colorCounts[c] = (colorCounts[c] || 0) + 1;
  });
  const topColor = Object.entries(colorCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "—";

  return (
    <div className="app-container">
      {/* ── Header ──────────────────────────────── */}
      <header className="header glass">
        <div className="logo">Style<span>Sphere</span></div>
        <div className="header-actions">
          <div className="badge">AI Pipeline Active</div>
        </div>
      </header>

      {/* ── Stats ───────────────────────────────── */}
      <div className="stats-row">
        <div className="stat-card glass">
          <div className="stat-value">{garments.length}</div>
          <div className="stat-label">Total Items</div>
        </div>
        <div className="stat-card glass">
          <div className="stat-value">{new Set(garments.map((g) => g.category)).size}</div>
          <div className="stat-label">Categories</div>
        </div>
        <div className="stat-card glass">
          <div className="stat-value">{Object.keys(colorCounts).length}</div>
          <div className="stat-label">Colors Detected</div>
        </div>
        <div className="stat-card glass">
          <div className="stat-value" style={{ fontSize: "1.3rem" }}>{topColor}</div>
          <div className="stat-label">Dominant Color</div>
        </div>
      </div>

      {/* ── Upload ──────────────────────────────── */}
      <section style={{ marginBottom: 40 }}>
        <div className="section-header">
          <h2 className="section-title"><span className="icon">📤</span> Upload Garment</h2>
        </div>
        <UploadZone onUploadComplete={handleUploadComplete} />
      </section>

      {/* ── Wardrobe ────────────────────────────── */}
      <section style={{ marginBottom: 60 }}>
        <div className="section-header">
          <h2 className="section-title"><span className="icon">👗</span> Virtual Wardrobe</h2>
        </div>
        <div className="filter-bar" style={{ marginBottom: 20 }}>
          {CATEGORIES.map((c) => (
            <button
              key={c.value}
              className={`filter-chip ${filter === c.value ? "active" : ""}`}
              onClick={() => setFilter(c.value)}
            >
              {c.label}
            </button>
          ))}
        </div>
        <WardrobeGrid garments={garments} filter={filter} onDelete={handleDelete} onTryOn={handleTryOn} />
      </section>

      {/* ── Try-On Mode ─────────────────────────── */}
      <section style={{ marginBottom: 60 }} className="tryon-section">
        <div className="section-header">
          <h2 className="section-title"><span className="icon">📸</span> Try-On Mode</h2>
          <p>Upload a photo of yourself, then click the ✨ icon on any garment to see how it looks!</p>
        </div>
        
        <div className="tryon-container glass" style={{ padding: "2rem", borderRadius: "16px", display: "flex", gap: "2rem", flexWrap: "wrap" }}>
          <div className="user-photo-zone" style={{ flex: "1 1 300px" }}>
            <h3>1. Base Photo</h3>
            <div className="upload-box" style={{ marginTop: 15, position: "relative" }}>
              <input 
                type="file" 
                accept="image/*" 
                onChange={handleUserPhotoUpload} 
                style={{ position: "absolute", inset: 0, opacity: 0, cursor: "pointer", zIndex: 2 }} 
              />
              {userPhotoUrl ? (
                <img src={userPhotoUrl} alt="User Base" style={{ width: "100%", height: "auto", borderRadius: "8px", display: "block" }} />
              ) : (
                <div style={{ padding: "3rem 1rem", textAlign: "center", border: "2px dashed var(--border)", borderRadius: "8px" }}>
                  <div style={{ fontSize: "2rem", marginBottom: "1rem" }}>👤</div>
                  <div>Click or drag to upload your photo</div>
                </div>
              )}
            </div>
          </div>

          <div className="tryon-result-zone" style={{ flex: "1 1 300px" }}>
            <h3>2. Live Preview</h3>
            <div className="result-box" style={{ marginTop: 15, padding: "1rem", border: "2px dashed var(--border)", borderRadius: "8px", minHeight: "250px", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", position: "relative" }}>
              {isTryingOn ? (
                <div className="loader-container" style={{ textAlign: "center" }}>
                  <div className="loader" style={{ fontSize: "2rem", animation: "spin 2s linear infinite" }}>🧠</div>
                  <div style={{ marginTop: 10, fontWeight: "bold" }}>AI Diffusion Model is generating...</div>
                  <div style={{ marginTop: 5, fontSize: "0.85rem", opacity: 0.7 }}>This takes ~60 seconds. Please wait.</div>
                </div>
              ) : tryOnResultUrl ? (
                <>
                  <img src={tryOnResultUrl} alt="Try-On Result" style={{ width: "100%", height: "auto", borderRadius: "8px", display: "block" }} />
                  {styleFeedback && (
                    <div className="feedback-badge" style={{ marginTop: "1rem", padding: "0.8rem", background: "var(--primary-light)", color: "var(--primary)", borderRadius: "8px", fontWeight: "bold", textAlign: "center", width: "100%" }}>
                      💡 {styleFeedback}
                    </div>
                  )}
                </>
              ) : (
                <div style={{ textAlign: "center", opacity: 0.6 }}>
                  <div style={{ fontSize: "2rem", marginBottom: "1rem" }}>✨</div>
                  <div>Select a garment to try on</div>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ── Toast ───────────────────────────────── */}
      {toast && <div className={`toast ${toast.type}`}>{toast.msg}</div>}
    </div>
  );
}
