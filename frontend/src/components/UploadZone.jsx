import { useState, useRef } from "react";
import { uploadGarment } from "../api";

export default function UploadZone({ onUploadComplete }) {
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [step, setStep] = useState({ current: 0, total: 5, label: "" });
  const [name, setName] = useState("");
  const [category, setCategory] = useState("other");
  const inputRef = useRef();

  const handleFile = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setName(file.name.replace(/\.[^.]+$/, ""));
    setPreview(URL.createObjectURL(file));
    setShowModal(true);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const startUpload = async () => {
    if (!selectedFile) return;
    setShowModal(false);
    setUploading(true);

    try {
      const result = await uploadGarment(selectedFile, name, category, (cur, total, label) => {
        setStep({ current: cur, total, label });
      });
      onUploadComplete?.(result);
    } catch (err) {
      console.error(err);
    } finally {
      setTimeout(() => {
        setUploading(false);
        setStep({ current: 0, total: 5, label: "" });
        setSelectedFile(null);
        setPreview(null);
      }, 1200);
    }
  };

  const pipelineSteps = [
    { num: "01", label: "Validate" },
    { num: "02", label: "AI Background Removal" },
    { num: "03", label: "Color Extract" },
    { num: "04", label: "Store" },
  ];

  const progress = step.total > 0 ? (step.current / step.total) * 100 : 0;

  return (
    <>
      <div
        className={`upload-zone glass ${dragOver ? "drag-over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !uploading && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          onChange={(e) => handleFile(e.target.files[0])}
        />
        {!uploading ? (
          <>
            <div className="upload-icon">✨</div>
            <h3>Drop your garment image here</h3>
            <p>Supports JPG, PNG, WebP, BMP — or click to browse</p>
            <div style={{ marginTop: 12, padding: "10px 16px", background: "rgba(255,193,7,0.1)", border: "1px solid rgba(255,193,7,0.3)", borderRadius: 8, fontSize: "0.82rem", color: "var(--text-secondary)", maxWidth: 420, margin: "12px auto 0" }}>
              <strong>📌 Best results tip:</strong> Use garment photos on a <strong>hanger</strong> or <strong>flat-lay</strong> against a <strong>plain/white background</strong>. Avoid photos with models — their hands and body may distort the try-on result.
            </div>
          </>
        ) : (
          <>
            <div className="upload-icon">⚙️</div>
            <h3>Processing through DIP pipeline…</h3>
            <p>{step.label}</p>
            <div className="pipeline-tracker">
              {pipelineSteps.map((s, i) => (
                <div
                  key={i}
                  className={`pipeline-step ${i < step.current ? "done" : ""} ${i === step.current ? "active" : ""}`}
                >
                  <span className="step-num">{i < step.current ? "✓" : s.num}</span>
                  {s.label}
                </div>
              ))}
            </div>
            <div className="progress-bar-container">
              <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
            </div>
          </>
        )}
      </div>

      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal glass" onClick={(e) => e.stopPropagation()}>
            <h2>🏷️ Add Garment Details</h2>
            {preview && (
              <div style={{ marginBottom: 16, borderRadius: 12, overflow: "hidden", maxHeight: 200 }}>
                <img src={preview} alt="preview" style={{ width: "100%", objectFit: "cover", maxHeight: 200 }} />
              </div>
            )}
            <div className="form-group">
              <label>Garment Name</label>
              <input className="form-input" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Summer Floral Dress" />
            </div>
            <div className="form-group">
              <label>Category</label>
              <select className="form-select" value={category} onChange={(e) => setCategory(e.target.value)}>
                <option value="top">Top</option>
                <option value="bottom">Bottom</option>
                <option value="dress">Dress / Suit / Full Outfit</option>
                <option value="shoes">Shoes</option>
                <option value="accessory">Accessory</option>
                <option value="other">Other</option>
              </select>
            </div>
            <div className="form-actions">
              <button className="btn btn-ghost" onClick={() => setShowModal(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={startUpload}>✨ Process & Add</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
