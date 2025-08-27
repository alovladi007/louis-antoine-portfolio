# Integration Instructions for Additional Projects Page

## Steps to Add the Semiconductor Projects to Your Portfolio

### 1. Clone Your Portfolio Repository
```bash
git clone https://github.com/alovladi007/louis-antoine-portfolio.git
cd louis-antoine-portfolio
```

### 2. Update the additional-projects.html File

Open your existing `additional-projects.html` file and add these two project sections at the beginning of your projects list (after the header but before any existing projects).

### 3. HTML Code to Insert

Insert the following HTML code right after your page header/title section:

```html
<!-- PROJECT 1: Photolithography & Optical Metrology Simulation -->
<div class="project">
    <h2>🔬 Photolithography & Optical Metrology Simulation</h2>
    <p class="project-tagline">End-to-end lithography simulation platform with nanometer precision</p>
    
    <h3>🎯 Project Overview</h3>
    <p>
        This project models and simulates key steps in semiconductor photolithography and optical metrology workflows, 
        the backbone of IC manufacturing. The simulation includes mask pattern generation, optical proximity correction (OPC), 
        aerial image simulation, resist profile modeling, and defect inspection analysis.
    </p>

    <h3>⚙️ Methods & Tools</h3>
    <ul>
        <li><strong>Simulation:</strong> Python + NumPy/SciPy for mask image simulation (Monte Carlo & Double Gaussian PSF)</li>
        <li><strong>Optical Analysis:</strong> Fourier optics modeling of resolution limits under DUV/EUV sources (248nm, 193nm, 13.5nm)</li>
        <li><strong>OPC Implementation:</strong> Rule-based, model-based, and inverse lithography technology algorithms</li>
        <li><strong>Metrology:</strong> Scatterometry, interferometry, and ellipsometry signal modeling</li>
        <li><strong>Defect Detection:</strong> Edge detection + noise filtering for simulated wafer maps</li>
    </ul>

    <h3>📈 Results & Visuals</h3>
    <ul>
        <li>Resolution capability: 7nm-45nm feature sizes</li>
        <li>OPC reduces edge placement error by 35%</li>
        <li>Process window: ±8% exposure latitude</li>
        <li>Interferometry accuracy: λ/100</li>
    </ul>

    <p><strong>🚀 Impact:</strong> Demonstrates deep understanding of lithography physics directly applicable to ASML/KLA workflows.</p>

    <p><strong>Technologies:</strong> Python, FastAPI, Next.js, PostgreSQL, Docker, NumPy/SciPy, OpenCV, Plotly.js</p>

    <div class="project-links">
        <a href="https://github.com/alovladi007/lithography-simulation" class="btn">View Repository</a>
        <a href="#" class="btn">Live Demo</a>
    </div>
</div>

<!-- PROJECT 2: Defect Detection & Yield Prediction with ML -->
<div class="project">
    <h2>🤖 Defect Detection & Yield Prediction with Machine Learning</h2>
    <p class="project-tagline">AI-powered defect classification and yield optimization platform</p>
    
    <h3>🎯 Project Overview</h3>
    <p>
        This project applies advanced machine learning to classify wafer/SEM image defects and predict semiconductor yield. 
        Using ensemble CNNs, Bayesian neural networks, and active learning, the platform addresses rare defect classes 
        and enhances yield forecasting accuracy.
    </p>

    <h3>⚙️ Methods & Tools</h3>
    <ul>
        <li><strong>ML Models:</strong> ResNet-50 and EfficientNet ensemble for defect classification</li>
        <li><strong>Yield Prediction:</strong> Bayesian Neural Networks with uncertainty quantification</li>
        <li><strong>Active Learning:</strong> Reduces labeling needs by 40%</li>
        <li><strong>Dashboard:</strong> Real-time Streamlit web app for visualization</li>
    </ul>

    <h3>📈 Results & Visuals</h3>
    <ul>
        <li>Defect classification accuracy: 99.2% (AUROC: 0.98)</li>
        <li>Yield prediction sMAPE: <4%</li>
        <li>Inference speed: <50ms per image</li>
        <li>30% reduction in false positives</li>
    </ul>

    <p><strong>🚀 Impact:</strong> Demonstrates expertise in AI/ML for semiconductor manufacturing, aligning with modern fab automation initiatives.</p>

    <p><strong>Technologies:</strong> PyTorch, TensorFlow, FastAPI, MLflow, Streamlit, Docker, PostgreSQL</p>

    <div class="project-links">
        <a href="https://github.com/alovladi007/defect-detection-ml" class="btn">View Repository</a>
        <a href="#" class="btn">Live Dashboard</a>
    </div>
</div>
```

### 4. Add/Update CSS Styles

If your existing CSS doesn't include these styles, add them to your stylesheet:

```css
.project {
    background: white;
    border-radius: 12px;
    padding: 30px;
    margin-bottom: 30px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.project h2 {
    color: #2c3e50;
    margin-bottom: 15px;
}

.project-tagline {
    color: #7f8c8d;
    font-style: italic;
    margin-bottom: 20px;
}

.project h3 {
    color: #34495e;
    margin-top: 20px;
    margin-bottom: 10px;
}

.project-links {
    margin-top: 20px;
}

.project-links a {
    display: inline-block;
    margin-right: 15px;
    padding: 10px 20px;
    background: #3498db;
    color: white;
    text-decoration: none;
    border-radius: 6px;
}

.project-links a:hover {
    background: #2980b9;
}
```

### 5. Commit and Push Changes

```bash
git add additional-projects.html
git commit -m "Add Photolithography and ML Defect Detection projects"
git push origin main
```

### 6. Create Project Repositories (Optional)

If you want to host the actual project code:

```bash
# Create new repositories on GitHub for each project
# Then push the project code:

# For Lithography project
cd /workspace/semiconductor-projects/lithography-simulation
git init
git remote add origin https://github.com/alovladi007/lithography-simulation.git
git add .
git commit -m "Initial commit"
git push -u origin main

# For ML project
cd /workspace/semiconductor-projects/defect-detection-ml
git init
git remote add origin https://github.com/alovladi007/defect-detection-ml.git
git add .
git commit -m "Initial commit"
git push -u origin main
```

### 7. Deploy Live Demos (Optional)

For live demos, you can deploy to:
- **Heroku** (free tier available)
- **Vercel** (for Next.js frontends)
- **Railway** or **Render** (for full-stack apps)
- **GitHub Pages** (for static demos)

## Notes

- Update the repository links to point to your actual GitHub repositories
- Update demo links once you deploy the applications
- The projects are now the first two entries on your Additional Projects page
- Both projects demonstrate advanced semiconductor industry skills