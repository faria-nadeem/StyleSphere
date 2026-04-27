# ✨ StyleSphere — Virtual Wardrobe & AI Try-On

A full-stack web application that combines **Digital Image Processing (DIP)** with an **AI-powered pipeline** to let users upload garment images, automatically remove backgrounds, detect dominant colours, and manage a virtual wardrobe.

---

## 🏗️ Architecture

```
stylesphere/
├── backend/                   # Python / FastAPI
│   ├── main.py                # App entry point
│   ├── database.py            # SQLAlchemy + SQLite setup
│   ├── models.py              # User & Garment ORM models
│   ├── requirements.txt       # Python dependencies
│   ├── routes/
│   │   └── garments.py        # Upload, list, get, delete endpoints
│   ├── services/
│   │   ├── image_processing.py  # Gaussian Blur → GrabCut → HSV Histogram
│   │   └── ai_pipeline.py      # Pose Estimation & Body Segmentation stubs
│   └── uploads/               # Processed images (auto-created)
├── frontend/                  # React + Vite
│   ├── src/
│   │   ├── App.jsx            # Main application shell
│   │   ├── api.js             # Backend API client
│   │   ├── index.css          # Full design system (glassmorphism)
│   │   └── components/
│   │       ├── UploadZone.jsx   # Drag-and-drop + pipeline progress
│   │       └── WardrobeGrid.jsx # Filterable garment grid
│   └── index.html
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

| Tool   | Version |
|--------|---------|
| Python | 3.10+   |
| Node   | 18+     |
| npm    | 9+      |

### 1. Backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload --port 8000
```

The API will be available at **http://localhost:8000** — interactive docs at `/docs`.

### 2. Frontend

```bash
cd frontend

npm install
npm run dev
```

The app will open at **http://localhost:5173**.

---

## 🔬 DIP Pipeline

Each uploaded image goes through the following pipeline:

| Step | Technique | Purpose |
|------|-----------|---------|
| 1 | **Gaussian Blur** | Noise reduction before segmentation |
| 2 | **GrabCut** (OpenCV) | Foreground/background separation |
| 3 | **HSV Histogram** | Dominant garment colour extraction |
| 4 | **Pose Estimation** *(optional)* | 33-point body landmark detection |
| 5 | **Body Segmentation** *(optional)* | Pixel-level body mask |

Steps 4 & 5 require **MediaPipe** and activate automatically if installed.

---

## 🎨 Design System

- **Palette:** Lavender `#E6E6FA`, Pastel Mint `#B8E8D0`, Creamy White `#FFF8F0`, Rose Quartz `#F2A7B3`
- **Style:** Glassmorphism with frosted-glass cards, 20px+ border-radius, soft shadows
- **Typography:** [Outfit](https://fonts.google.com/specimen/Outfit) (headings) + [Inter](https://fonts.google.com/specimen/Inter) (body)
- **Animations:** Floating icons, pulse badges, hover lift effects

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/garments/upload` | Upload image → run DIP pipeline → store |
| `GET`  | `/api/garments/` | List all garments (query: `user_id`) |
| `GET`  | `/api/garments/{id}` | Get garment details + features |
| `DELETE` | `/api/garments/{id}` | Remove a garment |
| `GET`  | `/api/health` | Health check |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Uvicorn, SQLAlchemy, SQLite |
| Image Processing | OpenCV (Gaussian Blur, GrabCut, HSV) |
| AI Models | MediaPipe Pose & Selfie Segmentation |
| Frontend | React 19, Vite 6 |
| Styling | Vanilla CSS with Glassmorphism design system |

---

## 📄 License

MIT — feel free to use and modify.
