# 🦠 Silent Outbreak Predictor

The Silent Outbreak Predictor — Early Warning Surveillance Dashboard is an AI-powered public health tool designed to detect infectious disease outbreaks before they hit hospitals.

Here is the system :
Pre-Diagnostic Data: It scans "silent" signals—like wastewater, social media chatter, mobility data, and search trends—rather than waiting for delayed clinical diagnoses.
AI Forecasting: It uses advanced machine learning models to predict the timing and trajectory of potential pathogens.
Geospatial Mapping: It pinpoints exact geographic hotspots to visualize where a disease is likely to cluster and spread.

---

Automated Alerts: It issues tiered warnings (Informational, Warning, Critical) to give health authorities a critical head start on containment and resource allocation.
**🌐 Live Demo Website:** **[https://silent-outbreak-predictor.onrender.com/](https://silent-outbreak-predictor.onrender.com/)**

---

## 🏗️ Architecture & Component Stack

The platform is designed as a modular, high-performance monorepo:

```
                  ┌──────────────────────────────────────────┐
                  │          Client Web Dashboard            │
                  │              (HTML/JS/CSS)               │
                  └────────────────────┬─────────────────────┘
                                       │ (Rest API / Auth)
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │             FastAPI Backend              │
                  │               (Python API)               │
                  └────────────────────┬─────────────────────┘
                                       │ (Orchestrates)
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │         CrewAI Pipeline Core             │
                  │             (Multi-Agent)                │
                  └──────┬─────────────┬─────────────┬───────┘
                         │             │             │
                         ▼             ▼             ▼
                  ┌────────────┐ ┌────────────┐ ┌────────────┐
                  │  Gatherer  │ │  Analyst   │ │ Visualizer │
                  │   Agent    │ │   Agent    │ │   Agent    │
                  └────────────┘ └────────────┘ └────────────┘
```

### 1. **[Frontend Dashboard](file:///c:/Users/VEDAN/Silent-Outbreak-Predictor/frontend/index.html)**
* **Visuals:** Premium dark-mode user interface using Harmonious color palettes, glassmorphism, responsive grids, and subtle floating micro-animations.
* **Charts:** Real-time canvas integrations rendering source distributions, confidence breakdowns, category donuts, and historic risk trends.
* **Authentication:** A hybrid landing page that seamlessly uses **Firebase Authentication** if configured, or falls back to a persistent `localStorage` mock database automatically.

### 2. **[Backend API Server](file:///c:/Users/VEDAN/Silent-Outbreak-Predictor/backend/api.py)**
* Powered by **FastAPI** to manage asynchronous analysis pipelines, track background worker statuses, and serve generated heatmap visualizations.
* Implements connection locking (`threading.Lock`) to prevent CrewAI concurrent execution crashes on Windows environments.

### 3. **[AI Core Pipeline](file:///c:/Users/VEDAN/Silent-Outbreak-Predictor/ai_core)**
* Powered by **CrewAI** with a three-agent sequential workflow:
  1. **OSINT Gatherer Agent** (`gpt-4o-mini`): Scrapes official bulletins (WHO, CDC, local health offices) and localized digital news.
  2. **Epidemiological Analyst Agent** (`gpt-4o`): Calculates traceable epidemic confidence scores (0–100%) and explains reasoning.
  3. **Visualizer Agent** (`gpt-4o-mini`): Generates custom, publication-quality risk charts on the local disk using Matplotlib.

---

## ⚡ Key Features

* **🔒 Hybrid Authentication:** Secure, password-validated Sign In and Sign Up page. Supports out-of-the-box integration with Firebase Auth while preserving server-side keys.
* **💾 Scoped Session Persistence:** Saves, displays, and restores historic reports dynamically under a separate key namespace for each logged-in user account.
* **🎛️ Dynamic LLM Settings:** Direct dropdown overrides to swap analysis models (`gpt-4o`, `gpt-4o-mini`, `claude-3-5-sonnet`, `o1-mini`) on the fly.
* **📡 Filtered Search Strategies:** Formulates custom CrewAI OSINT scraping strategies matching exactly the checked data sources (Government, Hospitals, Social Media, Local News).

---

## 🚀 Installation & Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/mfbvedant/Silent-Outbreak-Predictor.git
cd Silent-Outbreak-Predictor
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r ai_core/requirements.txt
pip install -r backend/requirements.txt
```

### 4. Configure Environment Variables
Create a file named `.env` inside the [ai_core/](file:///c:/Users/VEDAN/Silent-Outbreak-Predictor/ai_core) directory:
```env
OPENAI_API_KEY=your_openai_api_key

# Firebase Integration (Optional - falls back to local storage if empty)
FIREBASE_PROJECT_ID=your_firebase_project_id
FIREBASE_APP_ID=your_firebase_app_id
FIREBASE_MESSAGING_SENDER_ID=your_firebase_sender_id
FIREBASE_API_KEY=your_firebase_api_key
```

### 5. Launch the application
```bash
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```
Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your browser to access the dashboard!

---

## ☁️ Deployment Guide (Render & Cloud Run)

The application includes a [Dockerfile](file:///c:/Users/VEDAN/Silent-Outbreak-Predictor/Dockerfile) so it can be deployed live to any containerized hosting service.

### Deploying to Render (Free Hosting)
1. Sign up on **[Render.com](https://render.com/)** using your GitHub account.
2. Create a new **Web Service** and connect this repository.
3. Configure the settings:
   * **Runtime:** `Docker`
   * **Instance Type:** `Free`
4. Click **Advanced** and add your environment variables (`OPENAI_API_KEY`, `FIREBASE_API_KEY`, etc.).
5. Click **Deploy Web Service**! Render will build and host your app live.
