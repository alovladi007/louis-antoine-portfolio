from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
import numpy as np
import io
import json

from database import get_db, engine, Base
from models import SimulationModel, UserModel, ProjectModel
from schemas import (
    SimulationCreate, SimulationResponse, SimulationUpdate,
    UserCreate, UserResponse, ProjectCreate, ProjectResponse,
    LithographyParams, MetrologyParams, OPCParams
)
from auth import get_current_user, create_access_token, verify_password, get_password_hash
from simulation_engine import (
    LithographySimulator, 
    OpticalMetrology,
    OPCProcessor,
    DefectDetector
)
from celery_worker import celery_app, run_simulation_task
import redis
import pickle

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize Redis
redis_client = redis.Redis(host='redis', port=6379, decode_responses=False)

app = FastAPI(
    title="Photolithography & Optical Metrology Simulation API",
    description="Advanced semiconductor lithography simulation and metrology platform",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize simulation engines
litho_simulator = LithographySimulator()
metrology = OpticalMetrology()
opc_processor = OPCProcessor()
defect_detector = DefectDetector()

@app.get("/")
async def root():
    return {
        "message": "Photolithography & Optical Metrology Simulation API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "simulations": "/api/simulations",
            "lithography": "/api/lithography",
            "metrology": "/api/metrology",
            "opc": "/api/opc",
            "defects": "/api/defects"
        }
    }

@app.post("/api/auth/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    db_user = db.query(UserModel).filter(UserModel.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = UserModel(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password,
        full_name=user.full_name
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    access_token = create_access_token(data={"sub": user.email})
    return UserResponse(
        id=db_user.id,
        email=db_user.email,
        username=db_user.username,
        full_name=db_user.full_name,
        access_token=access_token
    )

@app.post("/api/lithography/simulate")
async def simulate_lithography(
    params: LithographyParams,
    background_tasks: BackgroundTasks,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Run photolithography simulation with given parameters"""
    
    # Create simulation record
    simulation = SimulationModel(
        name=params.name,
        type="lithography",
        user_id=current_user.id,
        parameters=params.dict(),
        status="pending"
    )
    db.add(simulation)
    db.commit()
    db.refresh(simulation)
    
    # Queue background task
    task = run_simulation_task.delay(simulation.id, "lithography", params.dict())
    
    # Store task ID in Redis
    redis_client.set(f"sim:{simulation.id}:task", task.id)
    
    return {
        "simulation_id": simulation.id,
        "task_id": task.id,
        "status": "queued",
        "message": "Lithography simulation started"
    }

@app.get("/api/lithography/results/{simulation_id}")
async def get_lithography_results(
    simulation_id: int,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get lithography simulation results"""
    
    simulation = db.query(SimulationModel).filter(
        SimulationModel.id == simulation_id,
        SimulationModel.user_id == current_user.id
    ).first()
    
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    # Get results from Redis cache
    cached_results = redis_client.get(f"sim:{simulation_id}:results")
    
    if cached_results:
        results = pickle.loads(cached_results)
    else:
        # Run simulation if not cached
        results = litho_simulator.run_simulation(simulation.parameters)
        # Cache results
        redis_client.setex(
            f"sim:{simulation_id}:results",
            3600,  # 1 hour TTL
            pickle.dumps(results)
        )
    
    return {
        "simulation_id": simulation_id,
        "status": simulation.status,
        "results": results,
        "parameters": simulation.parameters,
        "created_at": simulation.created_at
    }

@app.post("/api/lithography/mask-generation")
async def generate_mask_pattern(
    pattern_type: str = "line_space",
    feature_size: float = 45.0,  # nm
    pitch: float = 90.0,  # nm
    dimensions: List[int] = [1024, 1024],
    defects: bool = False,
    noise_level: float = 0.01
):
    """Generate mask pattern for lithography"""
    
    mask = litho_simulator.generate_mask(
        pattern_type=pattern_type,
        feature_size=feature_size,
        pitch=pitch,
        dimensions=dimensions,
        add_defects=defects,
        noise_level=noise_level
    )
    
    # Convert to image
    img_buffer = io.BytesIO()
    litho_simulator.save_mask_image(mask, img_buffer)
    img_buffer.seek(0)
    
    return StreamingResponse(
        img_buffer,
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename=mask_{pattern_type}.png"}
    )

@app.post("/api/opc/process")
async def apply_opc(
    params: OPCParams,
    mask_file: UploadFile = File(...),
    current_user: UserModel = Depends(get_current_user)
):
    """Apply Optical Proximity Correction to mask pattern"""
    
    # Read mask image
    mask_data = await mask_file.read()
    mask_array = opc_processor.load_mask(mask_data)
    
    # Apply OPC
    corrected_mask = opc_processor.apply_opc(
        mask_array,
        correction_type=params.correction_type,
        iterations=params.iterations,
        threshold=params.threshold
    )
    
    # Calculate metrics
    metrics = opc_processor.calculate_metrics(mask_array, corrected_mask)
    
    # Generate comparison image
    comparison = opc_processor.generate_comparison(mask_array, corrected_mask)
    
    img_buffer = io.BytesIO()
    opc_processor.save_comparison(comparison, img_buffer)
    img_buffer.seek(0)
    
    return {
        "metrics": metrics,
        "improvement": {
            "edge_placement_error": f"{metrics['epe_improvement']:.2f}%",
            "line_edge_roughness": f"{metrics['ler_improvement']:.2f}%",
            "corner_rounding": f"{metrics['corner_improvement']:.2f}%"
        },
        "comparison_image": f"data:image/png;base64,{opc_processor.encode_image(img_buffer)}"
    }

@app.post("/api/metrology/measure")
async def perform_metrology(
    params: MetrologyParams,
    wafer_image: UploadFile = File(...),
    current_user: UserModel = Depends(get_current_user)
):
    """Perform optical metrology measurements"""
    
    # Read wafer image
    image_data = await wafer_image.read()
    
    # Perform measurements based on technique
    if params.technique == "scatterometry":
        results = metrology.scatterometry_analysis(
            image_data,
            wavelength=params.wavelength,
            angle_range=params.angle_range,
            polarization=params.polarization
        )
    elif params.technique == "interferometry":
        results = metrology.interferometry_measurement(
            image_data,
            wavelength=params.wavelength,
            reference_mirror=params.reference_mirror
        )
    elif params.technique == "ellipsometry":
        results = metrology.ellipsometry_analysis(
            image_data,
            wavelength_range=params.wavelength_range,
            incident_angle=params.incident_angle
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid metrology technique")
    
    return {
        "technique": params.technique,
        "measurements": results["measurements"],
        "uncertainty": results["uncertainty"],
        "visualization": results["plots"],
        "quality_metrics": results["quality"]
    }

@app.post("/api/defects/detect")
async def detect_defects(
    wafer_image: UploadFile = File(...),
    sensitivity: float = 0.95,
    min_defect_size: int = 5,
    classification: bool = True
):
    """Detect and classify defects in wafer images"""
    
    # Read image
    image_data = await wafer_image.read()
    
    # Detect defects
    defects = defect_detector.detect(
        image_data,
        sensitivity=sensitivity,
        min_size=min_defect_size
    )
    
    # Classify if requested
    if classification and defects:
        classifications = defect_detector.classify_defects(defects)
        for defect, cls in zip(defects, classifications):
            defect["classification"] = cls
    
    # Generate defect map
    defect_map = defect_detector.generate_defect_map(image_data, defects)
    
    return {
        "total_defects": len(defects),
        "defects": defects,
        "defect_density": len(defects) / (1024 * 1024) * 1e6,  # defects per mm²
        "defect_map": defect_map,
        "statistics": defect_detector.calculate_statistics(defects)
    }

@app.get("/api/simulations", response_model=List[SimulationResponse])
async def list_simulations(
    skip: int = 0,
    limit: int = 100,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all simulations for current user"""
    
    simulations = db.query(SimulationModel).filter(
        SimulationModel.user_id == current_user.id
    ).offset(skip).limit(limit).all()
    
    return [SimulationResponse.from_orm(sim) for sim in simulations]

@app.get("/api/simulations/{simulation_id}", response_model=SimulationResponse)
async def get_simulation(
    simulation_id: int,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific simulation details"""
    
    simulation = db.query(SimulationModel).filter(
        SimulationModel.id == simulation_id,
        SimulationModel.user_id == current_user.id
    ).first()
    
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    return SimulationResponse.from_orm(simulation)

@app.delete("/api/simulations/{simulation_id}")
async def delete_simulation(
    simulation_id: int,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a simulation"""
    
    simulation = db.query(SimulationModel).filter(
        SimulationModel.id == simulation_id,
        SimulationModel.user_id == current_user.id
    ).first()
    
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    # Clean up cached data
    redis_client.delete(f"sim:{simulation_id}:results")
    redis_client.delete(f"sim:{simulation_id}:task")
    
    db.delete(simulation)
    db.commit()
    
    return {"message": "Simulation deleted successfully"}

@app.post("/api/projects", response_model=ProjectResponse)
async def create_project(
    project: ProjectCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new project"""
    
    db_project = ProjectModel(
        **project.dict(),
        user_id=current_user.id
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    
    return ProjectResponse.from_orm(db_project)

@app.get("/api/projects", response_model=List[ProjectResponse])
async def list_projects(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all projects for current user"""
    
    projects = db.query(ProjectModel).filter(
        ProjectModel.user_id == current_user.id
    ).all()
    
    return [ProjectResponse.from_orm(proj) for proj in projects]

@app.get("/api/wavelengths")
async def get_wavelength_options():
    """Get available wavelength options for lithography"""
    return {
        "duv": {
            "248nm": {"name": "KrF", "resolution": 140, "k1": 0.6},
            "193nm": {"name": "ArF", "resolution": 90, "k1": 0.5},
            "193nm_immersion": {"name": "ArF Immersion", "resolution": 45, "k1": 0.4}
        },
        "euv": {
            "13.5nm": {"name": "EUV", "resolution": 18, "k1": 0.35},
            "6.7nm": {"name": "Beyond EUV", "resolution": 8, "k1": 0.3}
        }
    }

@app.get("/api/patterns")
async def get_pattern_library():
    """Get available mask pattern templates"""
    return {
        "patterns": [
            {"id": "line_space", "name": "Line/Space", "min_feature": 20},
            {"id": "contact_hole", "name": "Contact Holes", "min_feature": 30},
            {"id": "sram", "name": "SRAM Cell", "min_feature": 25},
            {"id": "logic", "name": "Logic Gate", "min_feature": 22},
            {"id": "test_pattern", "name": "Test Pattern", "min_feature": 15}
        ]
    }

@app.websocket("/ws/simulation/{simulation_id}")
async def websocket_simulation_updates(websocket, simulation_id: int):
    """WebSocket for real-time simulation updates"""
    await websocket.accept()
    
    try:
        while True:
            # Get simulation status from Redis
            status = redis_client.get(f"sim:{simulation_id}:status")
            progress = redis_client.get(f"sim:{simulation_id}:progress")
            
            if status:
                await websocket.send_json({
                    "status": status.decode() if status else "unknown",
                    "progress": float(progress.decode()) if progress else 0,
                    "simulation_id": simulation_id
                })
            
            if status and status.decode() in ["completed", "failed"]:
                break
                
            await asyncio.sleep(1)
            
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)