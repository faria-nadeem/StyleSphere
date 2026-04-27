import { useState, useEffect, useCallback } from "react";
import UploadZone from "./components/UploadZone";
import WardrobeGrid from "./components/WardrobeGrid";
import { fetchGarments, deleteGarment } from "./api";
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
        <WardrobeGrid garments={garments} filter={filter} onDelete={handleDelete} />
      </section>

      {/* ── Toast ───────────────────────────────── */}
      {toast && <div className={`toast ${toast.type}`}>{toast.msg}</div>}
    </div>
  );
}
