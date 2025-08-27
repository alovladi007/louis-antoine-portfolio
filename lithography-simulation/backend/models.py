from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class UserModel(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    simulations = relationship("SimulationModel", back_populates="user")
    projects = relationship("ProjectModel", back_populates="user")

class SimulationModel(Base):
    __tablename__ = "simulations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # lithography, metrology, opc, defect
    status = Column(String, default="pending")  # pending, running, completed, failed
    parameters = Column(JSON, nullable=False)
    results = Column(JSON)
    error_message = Column(Text)
    user_id = Column(Integer, ForeignKey("users.id"))
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    user = relationship("UserModel", back_populates="simulations")
    project = relationship("ProjectModel", back_populates="simulations")

class ProjectModel(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    type = Column(String)  # research, production, development
    status = Column(String, default="active")
    metadata = Column(JSON)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    user = relationship("UserModel", back_populates="projects")
    simulations = relationship("SimulationModel", back_populates="project")

class MaskPatternModel(Base):
    __tablename__ = "mask_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    pattern_type = Column(String, nullable=False)
    feature_size = Column(Float, nullable=False)
    pitch = Column(Float, nullable=False)
    dimensions = Column(JSON, nullable=False)
    file_path = Column(String)
    metadata = Column(JSON)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class MetrologyResultModel(Base):
    __tablename__ = "metrology_results"
    
    id = Column(Integer, primary_key=True, index=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    technique = Column(String, nullable=False)
    measurements = Column(JSON, nullable=False)
    uncertainty = Column(JSON)
    quality_score = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())