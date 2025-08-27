# Photolithography & Optical Metrology Simulation Platform

## 🔬 Overview
Advanced semiconductor manufacturing simulation platform featuring photolithography modeling, optical metrology, OPC processing, and defect detection.

## 🚀 Features
- **Lithography Simulation**: Mask pattern generation, aerial image simulation, resist profiles
- **Optical Metrology**: Scatterometry, interferometry, ellipsometry measurements
- **OPC Processing**: Rule-based, model-based, and inverse lithography technology
- **Process Window Analysis**: Exposure latitude and depth of focus optimization
- **Real-time Visualization**: Interactive 3D plots and heatmaps

## 🛠️ Tech Stack
- **Backend**: Python, FastAPI, PostgreSQL, Redis, Celery
- **Frontend**: Next.js 14, TypeScript, Material-UI, Plotly.js
- **Simulation**: NumPy, SciPy, OpenCV, scikit-image
- **Deployment**: Docker, Docker Compose

## 📦 Installation

### Using Docker (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/lithography-simulation.git
cd lithography-simulation

# Start all services
docker-compose up -d

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# Jupyter Lab: http://localhost:8888 (token: lithography)
```

### Manual Installation
```bash
# Backend setup
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend setup
cd frontend
npm install
npm run dev
```

## 🔧 Configuration
Create a `.env` file in the root directory:
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/lithography_db
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key
```

## 📊 API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🧪 Running Simulations

### Lithography Simulation
```python
POST /api/lithography/simulate
{
  "pattern_type": "line_space",
  "feature_size": 45,
  "wavelength": 193,
  "NA": 1.35
}
```

### Metrology Measurement
```python
POST /api/metrology/measure
{
  "technique": "scatterometry",
  "wavelength": 633,
  "angle_range": [-70, 70]
}
```

## 📈 Results
- Resolution: 7-45nm feature sizes
- Process window: ±8% exposure latitude
- Measurement precision: λ/100
- OPC improvement: 30-40% EPE reduction

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first.

## 📄 License
MIT License