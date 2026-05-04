import { getImageUrl } from "../api";

export default function WardrobeGrid({ garments, filter, onDelete, onTryOn }) {
  const filtered = filter === "all"
    ? garments
    : garments.filter((g) => g.category === filter);

  if (filtered.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon">👗</div>
        <h3>Your wardrobe is empty</h3>
        <p>Upload your first garment to get started!</p>
      </div>
    );
  }

  return (
    <div className="wardrobe-grid">
      {filtered.map((g) => (
        <div className="garment-card" key={g.id}>
          <div className="card-actions">
            {onTryOn && (
              <button className="btn-icon" onClick={() => onTryOn(g.id)} title="Try On" style={{ marginRight: 8, fontSize: "1.2rem" }}>✨</button>
            )}
            <button className="btn-icon" onClick={() => onDelete?.(g.id)} title="Remove">🗑</button>
          </div>
          <div className="card-image">
            {g.image_path ? (
              <img src={getImageUrl(g.image_path)} alt={g.name} loading="lazy" />
            ) : (
              "👚"
            )}
          </div>
          <div className="card-body">
            <div className="card-title">{g.name}</div>
            <div className="card-meta">
              <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span
                  className="color-dot"
                  style={{ background: g.dominant_color?.hex || "#888" }}
                  title={g.dominant_color?.name}
                />
                {g.dominant_color?.name || "—"}
              </span>
              <span className="card-category">{g.category}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
