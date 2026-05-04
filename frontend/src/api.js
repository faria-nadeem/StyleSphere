const API = "http://localhost:8000";

export async function uploadGarment(file, name, category, onProgress) {
  const steps = [
    "Validating image…",
    "Running AI background removal…",
    "Extracting color features…",
    "Storing to database…",
  ];

  const form = new FormData();
  form.append("file", file);
  form.append("name", name);
  form.append("category", category);
  form.append("user_id", "default-user");

  // Simulate step-by-step progress while the server works
  let currentStep = 0;
  const interval = setInterval(() => {
    if (currentStep < steps.length - 1) {
      currentStep++;
      onProgress?.(currentStep, steps.length, steps[currentStep]);
    }
  }, 800);

  onProgress?.(0, steps.length, steps[0]);

  try {
    const res = await fetch(`${API}/api/garments/upload`, {
      method: "POST",
      body: form,
    });
    clearInterval(interval);
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload failed");
    }
    onProgress?.(steps.length, steps.length, "Complete!");
    return await res.json();
  } catch (e) {
    clearInterval(interval);
    throw e;
  }
}

export async function fetchGarments() {
  const res = await fetch(`${API}/api/garments/?user_id=default-user`);
  if (!res.ok) throw new Error("Failed to fetch garments");
  return res.json();
}

export async function deleteGarment(id) {
  const res = await fetch(`${API}/api/garments/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete garment");
  return res.json();
}

export function getImageUrl(path) {
  if (!path) return null;
  const normalized = path.replace(/\\/g, "/");
  return `${API}/${normalized}`;
}

export async function tryOnGarment(garmentId, userPhotoFile) {
  const form = new FormData();
  form.append("garment_id", garmentId);
  form.append("user_image", userPhotoFile);

  // AI model can take up to 2 minutes — set a generous timeout
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 180000); // 3 min

  try {
    const res = await fetch(`${API}/api/tryon`, {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    clearTimeout(timeout);
    
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Try-On failed");
    }
    return await res.json();
  } catch (e) {
    clearTimeout(timeout);
    if (e.name === 'AbortError') {
      throw new Error("AI model took too long. The server might be busy — please try again.");
    }
    throw e;
  }
}
