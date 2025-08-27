from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

# User Schemas
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str]
    access_token: str
    
    class Config:
        orm_mode = True

# Simulation Schemas
class SimulationCreate(BaseModel):
    name: str
    type: str
    parameters: Dict[str, Any]
    project_id: Optional[int] = None

class SimulationUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None

class SimulationResponse(BaseModel):
    id: int
    name: str
    type: str
    status: str
    parameters: Dict[str, Any]
    results: Optional[Dict[str, Any]]
    error_message: Optional[str]
    user_id: int
    project_id: Optional[int]
    created_at: datetime
    completed_at: Optional[datetime]
    
    class Config:
        orm_mode = True

# Project Schemas
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    type: Optional[str] = "research"
    metadata: Optional[Dict[str, Any]] = {}

class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    type: Optional[str]
    status: str
    metadata: Optional[Dict[str, Any]]
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        orm_mode = True

# Lithography Parameters
class LithographyParams(BaseModel):
    name: str = Field(default="Lithography Simulation")
    pattern_type: str = Field(default="line_space", description="Type of mask pattern")
    feature_size: float = Field(default=45.0, description="Feature size in nm")
    pitch: float = Field(default=90.0, description="Pattern pitch in nm")
    dimensions: List[int] = Field(default=[1024, 1024], description="Image dimensions")
    wavelength: float = Field(default=193.0, description="Exposure wavelength in nm")
    NA: float = Field(default=1.35, description="Numerical Aperture")
    sigma: float = Field(default=0.7, description="Partial coherence factor")
    defocus: float = Field(default=0.0, description="Defocus in nm")
    aberrations: Optional[Dict[str, float]] = Field(default={}, description="Optical aberrations")
    resist_thickness: float = Field(default=100.0, description="Resist thickness in nm")
    threshold: float = Field(default=0.3, description="Resist threshold")
    contrast: float = Field(default=5.0, description="Resist contrast")
    add_defects: bool = Field(default=False, description="Add mask defects")
    noise_level: float = Field(default=0.01, description="Noise level")
    exposure_range: Tuple[float, float] = Field(default=(0.8, 1.2))
    focus_range: Tuple[float, float] = Field(default=(-100, 100))

# OPC Parameters
class OPCParams(BaseModel):
    correction_type: str = Field(default="model_based", description="OPC algorithm type")
    iterations: int = Field(default=5, description="Number of iterations")
    threshold: float = Field(default=0.1, description="Convergence threshold")
    apply_mrc: bool = Field(default=True, description="Apply mask rule check")

# Metrology Parameters
class MetrologyParams(BaseModel):
    technique: str = Field(description="Metrology technique")
    wavelength: Optional[float] = Field(default=633.0, description="Wavelength in nm")
    wavelength_range: Optional[Tuple[float, float]] = Field(default=(400, 700))
    angle_range: Optional[Tuple[float, float]] = Field(default=(-70, 70))
    incident_angle: Optional[float] = Field(default=70.0)
    polarization: Optional[str] = Field(default="TE")
    reference_mirror: Optional[bool] = Field(default=True)

# Defect Detection Parameters
class DefectParams(BaseModel):
    sensitivity: float = Field(default=0.95, ge=0.0, le=1.0)
    min_defect_size: int = Field(default=5, ge=1)
    classification: bool = Field(default=True)

# Process Window Response
class ProcessWindowResponse(BaseModel):
    exposure_latitude: List[float]
    depth_of_focus: List[float]
    target_cd: float
    valid_window_percentage: float

# Defect Response
class DefectResponse(BaseModel):
    type: str
    x: int
    y: int
    width: int
    height: int
    area: int
    severity: float
    classification: Optional[Dict[str, Any]] = None

# Mask Pattern Schema
class MaskPatternCreate(BaseModel):
    name: str
    pattern_type: str
    feature_size: float
    pitch: float
    dimensions: List[int]
    metadata: Optional[Dict[str, Any]] = {}

class MaskPatternResponse(BaseModel):
    id: int
    name: str
    pattern_type: str
    feature_size: float
    pitch: float
    dimensions: List[int]
    file_path: Optional[str]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        orm_mode = True