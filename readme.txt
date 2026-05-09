=====================================
  STYLESPHERE - Setup Instructions
=====================================

StyleSphere is a virtual wardrobe app where you can upload clothes,
get skin-tone based recommendations, and try on outfits using AI.

Tech Stack:
  - Backend:  Python (FastAPI + SQLite + OpenCV + MediaPipe)
  - Frontend: React (Vite)
  - AI:       rembg (background removal), IDM-VTON & OOTDiffusion (virtual try-on via HuggingFace)


==============================
  WHAT YOU NEED INSTALLED
==============================

1. Python 3.10, 3.11, or 3.12  (NOT 3.13 - mediapipe doesnt support it yet)
   Download: https://www.python.org/downloads/

2. Node.js (v18 or newer)
   Download: https://nodejs.org/

3. Git (optional, only if cloning from GitHub)
   Download: https://git-scm.com/


==============================
  STEP 1: GET THE CODE
==============================

If you have the zip file:
  - Extract it to any folder on your PC

If cloning from GitHub:
  git clone https://github.com/faria-nadeem/StyleSphere.git
  cd StyleSphere
  git checkout feature/faria


==============================
  STEP 2: SETUP BACKEND
==============================

Open a terminal/command prompt and navigate to the backend folder:

  cd backend

Create a virtual environment:

  python -m venv venv

Activate it:
  - Windows (Command Prompt):   venv\Scripts\activate
  - Windows (PowerShell):       venv\Scripts\Activate.ps1
  - Mac/Linux:                  source venv/bin/activate

You should see (venv) at the start of your terminal line.

Install all the Python packages:

  pip install -r requirements.txt
  pip install rembg python-dotenv gradio_client

Note: rembg, python-dotenv and gradio_client are also required but
listed separately because they have large sub-dependencies.
The first time rembg runs, it will download a ~170MB AI model automatically.

Create the .env file in the backend folder:

  Create a file called .env and put this inside:
  HF_TOKEN=your_huggingface_token_here

  To get a free HuggingFace token:
  1. Go to https://huggingface.co and sign up
  2. Go to Settings > Access Tokens > New Token
  3. Copy the token and paste it in the .env file

  Note: The app will still work without a token, but virtual try-on
  will have limited usage (HuggingFace free GPU quota).


==============================
  STEP 3: SETUP FRONTEND
==============================

Open a NEW terminal (keep the backend one open) and navigate to frontend:

  cd frontend

Install the npm packages:

  npm install

Thats it for the frontend setup.


==============================
  STEP 4: RUN THE APP
==============================

You need TWO terminals running at the same time:

TERMINAL 1 - Backend (make sure venv is activated):

  cd backend
  venv\Scripts\activate        (or source venv/bin/activate on Mac/Linux)
  uvicorn main:app --reload --port 8000

  You should see:
    INFO:     Uvicorn running on http://127.0.0.1:8000
    INFO:     Application startup complete.

TERMINAL 2 - Frontend:

  cd frontend
  npm run dev

  You should see:
    VITE v8.x.x  ready
    Local:   http://localhost:5173/

Now open your browser and go to:

  http://localhost:5173

The app should be running!


==============================
  HOW TO USE THE APP
==============================

1. UPLOAD GARMENTS
   - Click the upload zone or drag-drop an image
   - Give it a name and pick a category (top, bottom, dress, etc.)
   - Click "Process & Add"
   - The app will remove the background and detect the dominant color

2. SKIN-TONE RECOMMENDATIONS
   - Scroll down to the "Skin-Tone Recommendations" section
   - Upload a selfie (clear, well-lit photo)
   - Click "Analyze My Skin Tone"
   - It will show your skin tone and which garments match best

3. VIRTUAL TRY-ON
   - Scroll down to "Try-On Mode"
   - Upload a front-facing photo of yourself in the "Base Photo" area
   - Go back to the wardrobe grid and click the sparkle button on any garment
   - Wait about 60 seconds for the AI to generate the result
   - The result appears in the "Live Preview" area

   Tips for best try-on results:
   - Use a front-facing photo with arms at your sides
   - Garment images work best on a hanger or flat-lay with white background
   - Avoid garment photos with models wearing them


==============================
  FOLDER STRUCTURE
==============================

StyleSphere/
  backend/
    main.py              - FastAPI entry point, try-on endpoint
    database.py          - SQLite database setup
    models.py            - User and Garment table definitions
    requirements.txt     - Python dependencies
    .env                 - HuggingFace token (you create this)
    routes/
      garments.py        - upload, list, delete garments
      recommendations.py - skin tone detection and recommendations
    services/
      image_processing.py - background removal and color extraction
      ai_pipeline.py      - pose estimation and body segmentation
      cv_tryon.py         - CV-based pants try-on (color transfer)
    uploads/             - where processed images are saved
    venv/                - Python virtual environment (created by you)

  frontend/
    src/
      App.jsx            - main React component
      api.js             - API calls to the backend
      index.css          - all the styles
      main.jsx           - React entry point
      components/
        UploadZone.jsx          - garment upload with drag-drop
        WardrobeGrid.jsx        - displays garment cards
        RecommendationPanel.jsx - skin tone analysis UI


==============================
  TROUBLESHOOTING
==============================

Problem: "Module not found" errors when starting backend
Fix:     Make sure your venv is activated and run:
         pip install -r requirements.txt
         pip install rembg python-dotenv gradio_client

Problem: MediaPipe errors on Python 3.13
Fix:     Use Python 3.12 or lower. MediaPipe doesnt support 3.13 yet.

Problem: Try-on gives "GPU quota exceeded" error
Fix:     The free HuggingFace tier has limited GPU usage.
         Wait a few minutes and try again, or sign up for a
         HuggingFace account and add your token to .env

Problem: Frontend shows blank page / network errors
Fix:     Make sure the backend is running on port 8000.
         Check that both terminals are still running.

Problem: "rembg" is very slow the first time
Fix:     Thats normal. It downloads a ~170MB model on first use.
         After that its cached and will be fast.

Problem: PowerShell says "running scripts is disabled"
Fix:     Run this command first:
         Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned


==============================
  CONTACT
==============================

If you have issues running this, reach out to the team.
GitHub: https://github.com/faria-nadeem/StyleSphere
