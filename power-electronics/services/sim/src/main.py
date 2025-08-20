"""
Power Electronics Simulation Service
Main FastAPI application with simulation endpoints
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import asyncio
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import structlog
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.core import CollectorRegistry
from starlette.responses import Response

from .kernels.interleaved_boost import InterleavedBoostSimulator
from .kernels.llc_resonant import LLCResonantSimulator
from .kernels.inverter import ThreePhaseInverterSimulator
from .kernels.pmsm_foc import PMSMFOCController
from .models.components import MOSFETModel, InductorModel, CapacitorModel
from .models.thermal import ThermalModel
from .utils.losses import calculate_switching_losses, calculate_conduction_losses

# Configure structured logging
logger = structlog.get_logger()

# Prometheus metrics
registry = CollectorRegistry()
sim_requests = Counter('sim_requests_total', 'Total simulation requests', registry=registry)
sim_duration = Histogram('sim_duration_seconds', 'Simulation duration', registry=registry)
active_sims = Gauge('active_simulations', 'Number of active simulations', registry=registry)
ws_connections = Gauge('websocket_connections', 'Active WebSocket connections', registry=registry)

# Redis client for caching and job queue
redis_client: Optional[redis.Redis] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global redis_client
    
    # Startup
    logger.info("Starting Power Electronics Simulation Service")
    redis_client = redis.from_url("redis://redis:6379", decode_responses=True)
    await redis_client.ping()
    logger.info("Connected to Redis")
    
    yield
    
    # Shutdown
    if redis_client:
        await redis_client.close()
    logger.info("Shutting down Simulation Service")


app = FastAPI(
    title="Power Electronics Simulation Service",
    description="Real-time simulation of power electronics topologies",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class SimulationRequest(BaseModel):
    """Simulation run request"""
    project_id: str
    topology: str = Field(..., pattern="^(interleaved_boost|llc_resonant|three_phase_inverter)$")
    parameters: Dict[str, Any]
    controller: Dict[str, Any]
    timespan: float = Field(0.01, gt=0, le=1.0, description="Simulation time in seconds")
    step_hint: float = Field(1e-6, gt=0, le=1e-3, description="Time step hint in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "project_id": "proj_123",
                "topology": "interleaved_boost",
                "parameters": {
                    "vin": 400,
                    "vout": 800,
                    "power": 10000,
                    "phases": 3,
                    "fsw": 100000,
                    "L": 100e-6,
                    "C": 1000e-6
                },
                "controller": {
                    "type": "current_mode",
                    "kp": 0.1,
                    "ki": 100
                },
                "timespan": 0.01,
                "step_hint": 1e-6
            }
        }


class SimulationResponse(BaseModel):
    """Simulation run response"""
    run_id: str
    status: str
    message: str
    
    
class SimulationStatus(BaseModel):
    """Simulation status"""
    run_id: str
    status: str
    progress: float
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class LossCalculationRequest(BaseModel):
    """Loss calculation request"""
    topology: str
    operating_point: Dict[str, float]
    device: Dict[str, Any]
    temperature: float = 25.0
    
    
class ThermalCalculationRequest(BaseModel):
    """Thermal calculation request"""
    power_loss: float
    ambient_temp: float = 25.0
    heatsink_rth: float = 0.5
    device_rth_jc: float = 0.3


# Simulation runners
async def run_interleaved_boost_simulation(
    run_id: str,
    params: Dict[str, Any],
    controller: Dict[str, Any],
    timespan: float,
    dt: float
) -> Dict[str, Any]:
    """Run interleaved boost converter simulation"""
    logger.info("Starting interleaved boost simulation", run_id=run_id)
    
    # Create simulator instance
    sim = InterleavedBoostSimulator(
        vin=params["vin"],
        vout_ref=params["vout"],
        power=params["power"],
        phases=params.get("phases", 3),
        fsw=params["fsw"],
        L=params["L"],
        C=params["C"],
        rload=params.get("rload", params["vout"]**2 / params["power"])
    )
    
    # Configure controller
    sim.set_controller(
        kp=controller.get("kp", 0.1),
        ki=controller.get("ki", 100),
        current_limit=controller.get("current_limit", 50)
    )
    
    # Run simulation
    results = await sim.simulate(timespan, dt)
    
    # Calculate losses
    mosfet = MOSFETModel(
        vds_max=1200,
        rds_on=0.045,
        qg=67e-9,
        coss=100e-12
    )
    
    switching_loss = calculate_switching_losses(
        mosfet,
        results["v_dc"].mean(),
        results["i_phases"].mean(axis=0),
        params["fsw"]
    )
    
    conduction_loss = calculate_conduction_losses(
        mosfet,
        results["i_phases"],
        results["duty_cycles"]
    )
    
    # Summary statistics
    summary = {
        "efficiency": results["efficiency"].mean(),
        "ripple_current": results["i_ripple"].max(),
        "output_ripple": results["v_out"].std(),
        "switching_loss": switching_loss,
        "conduction_loss": conduction_loss,
        "total_loss": switching_loss + conduction_loss,
        "thd": sim.calculate_thd(results["i_phases"])
    }
    
    # Store results in Redis
    await redis_client.hset(
        f"sim:{run_id}",
        mapping={
            "status": "completed",
            "progress": 100,
            "summary": str(summary),
            "waveforms": str(len(results["time"]))
        }
    )
    await redis_client.expire(f"sim:{run_id}", 3600)  # 1 hour TTL
    
    return {
        "summary": summary,
        "waveforms": {
            "time": results["time"].tolist()[:1000],  # Limit to 1000 points for response
            "v_out": results["v_out"].tolist()[:1000],
            "i_phases": results["i_phases"].tolist()[:1000],
            "duty_cycles": results["duty_cycles"].tolist()[:1000]
        }
    }


async def run_llc_resonant_simulation(
    run_id: str,
    params: Dict[str, Any],
    controller: Dict[str, Any],
    timespan: float,
    dt: float
) -> Dict[str, Any]:
    """Run LLC resonant converter simulation"""
    logger.info("Starting LLC resonant simulation", run_id=run_id)
    
    sim = LLCResonantSimulator(
        vin=params["vin"],
        vout_ref=params["vout"],
        power=params["power"],
        lr=params["lr"],
        cr=params["cr"],
        lm=params["lm"],
        n_ratio=params.get("n_ratio", 1.0)
    )
    
    sim.set_controller(
        kp=controller.get("kp", 0.01),
        ki=controller.get("ki", 10),
        freq_min=controller.get("freq_min", 50000),
        freq_max=controller.get("freq_max", 200000)
    )
    
    results = await sim.simulate(timespan, dt)
    
    # Check ZVS operation
    zvs_achieved = sim.check_zvs_operation(results)
    
    summary = {
        "efficiency": results["efficiency"].mean(),
        "zvs_achieved": zvs_achieved,
        "freq_average": results["frequency"].mean(),
        "gain": results["vout"].mean() / params["vin"],
        "output_ripple": results["vout"].std()
    }
    
    await redis_client.hset(
        f"sim:{run_id}",
        mapping={
            "status": "completed",
            "progress": 100,
            "summary": str(summary)
        }
    )
    
    return {
        "summary": summary,
        "waveforms": {
            "time": results["time"].tolist()[:1000],
            "vout": results["vout"].tolist()[:1000],
            "frequency": results["frequency"].tolist()[:1000],
            "tank_current": results["i_tank"].tolist()[:1000]
        }
    }


async def run_inverter_simulation(
    run_id: str,
    params: Dict[str, Any],
    controller: Dict[str, Any],
    timespan: float,
    dt: float
) -> Dict[str, Any]:
    """Run three-phase inverter simulation with optional motor control"""
    logger.info("Starting three-phase inverter simulation", run_id=run_id)
    
    # Create inverter simulator
    sim = ThreePhaseInverterSimulator(
        vdc=params["vdc"],
        fsw=params["fsw"],
        deadtime=params.get("deadtime", 1e-6),
        modulation=params.get("modulation", "SVPWM")
    )
    
    # If motor control is enabled, create FOC controller
    if params.get("motor_control", False):
        motor_params = params.get("motor", {})
        foc = PMSMFOCController(
            rs=motor_params.get("rs", 0.1),
            ld=motor_params.get("ld", 1e-3),
            lq=motor_params.get("lq", 1.5e-3),
            ke=motor_params.get("ke", 0.1),
            poles=motor_params.get("poles", 8),
            j_motor=motor_params.get("j", 0.01),
            b_friction=motor_params.get("b", 0.001)
        )
        
        foc.set_current_controller(
            kp_d=controller.get("kp_id", 10),
            ki_d=controller.get("ki_id", 100),
            kp_q=controller.get("kp_iq", 10),
            ki_q=controller.get("ki_iq", 100)
        )
        
        foc.set_speed_controller(
            kp=controller.get("kp_speed", 1),
            ki=controller.get("ki_speed", 10)
        )
        
        # Run combined simulation
        results = await sim.simulate_with_motor(foc, timespan, dt)
        
        summary = {
            "efficiency": results["efficiency"].mean(),
            "thd_current": sim.calculate_thd(results["i_abc"]),
            "torque_ripple": results["torque"].std() / results["torque"].mean(),
            "speed_error": abs(results["speed"][-1] - controller.get("speed_ref", 1000)) / 1000,
            "power_factor": results["power_factor"].mean()
        }
    else:
        # Grid-tie or standalone operation
        results = await sim.simulate(timespan, dt)
        
        summary = {
            "efficiency": results["efficiency"].mean(),
            "thd_voltage": sim.calculate_thd(results["v_abc"]),
            "thd_current": sim.calculate_thd(results["i_abc"]),
            "power_factor": results["power_factor"].mean(),
            "dc_link_ripple": results["vdc_ripple"].max()
        }
    
    await redis_client.hset(
        f"sim:{run_id}",
        mapping={
            "status": "completed",
            "progress": 100,
            "summary": str(summary)
        }
    )
    
    return {
        "summary": summary,
        "waveforms": {
            "time": results["time"].tolist()[:1000],
            "v_abc": results["v_abc"].tolist()[:1000],
            "i_abc": results["i_abc"].tolist()[:1000],
            "power": results["power"].tolist()[:1000]
        }
    }


# API Endpoints
@app.get("/")
async def root():
    """Health check and service info"""
    return {
        "service": "Power Electronics Simulation",
        "version": "1.0.0",
        "status": "healthy",
        "capabilities": [
            "interleaved_boost",
            "llc_resonant", 
            "three_phase_inverter",
            "pmsm_foc",
            "loss_calculation",
            "thermal_analysis"
        ]
    }


@app.post("/simulate/run", response_model=SimulationResponse)
async def start_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks
):
    """Start a new simulation run"""
    sim_requests.inc()
    run_id = str(uuid.uuid4())
    
    # Store initial status
    await redis_client.hset(
        f"sim:{run_id}",
        mapping={
            "status": "running",
            "progress": 0,
            "project_id": request.project_id,
            "topology": request.topology
        }
    )
    
    # Select appropriate simulator
    if request.topology == "interleaved_boost":
        background_tasks.add_task(
            run_interleaved_boost_simulation,
            run_id,
            request.parameters,
            request.controller,
            request.timespan,
            request.step_hint
        )
    elif request.topology == "llc_resonant":
        background_tasks.add_task(
            run_llc_resonant_simulation,
            run_id,
            request.parameters,
            request.controller,
            request.timespan,
            request.step_hint
        )
    elif request.topology == "three_phase_inverter":
        background_tasks.add_task(
            run_inverter_simulation,
            run_id,
            request.parameters,
            request.controller,
            request.timespan,
            request.step_hint
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown topology: {request.topology}")
    
    active_sims.inc()
    
    return SimulationResponse(
        run_id=run_id,
        status="started",
        message=f"Simulation {run_id} started for {request.topology}"
    )


@app.get("/simulate/run/{run_id}", response_model=SimulationStatus)
async def get_simulation_status(run_id: str):
    """Get simulation status and results"""
    data = await redis_client.hgetall(f"sim:{run_id}")
    
    if not data:
        raise HTTPException(status_code=404, detail=f"Simulation {run_id} not found")
    
    return SimulationStatus(
        run_id=run_id,
        status=data.get("status", "unknown"),
        progress=float(data.get("progress", 0)),
        summary=eval(data.get("summary", "{}")) if data.get("summary") else None,
        error=data.get("error")
    )


@app.websocket("/simulate/stream/{run_id}")
async def stream_simulation(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for streaming simulation results"""
    await websocket.accept()
    ws_connections.inc()
    
    try:
        while True:
            # Get current simulation data
            data = await redis_client.hgetall(f"sim:{run_id}")
            
            if not data:
                await websocket.send_json({"error": f"Simulation {run_id} not found"})
                break
            
            # Send status update
            await websocket.send_json({
                "status": data.get("status", "unknown"),
                "progress": float(data.get("progress", 0))
            })
            
            if data.get("status") == "completed":
                break
            
            await asyncio.sleep(0.1)  # Update rate
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", run_id=run_id)
    finally:
        ws_connections.dec()


@app.post("/calc/losses")
async def calculate_losses(request: LossCalculationRequest):
    """Calculate switching and conduction losses"""
    # Implementation depends on topology and device
    mosfet = MOSFETModel(**request.device)
    
    switching_loss = calculate_switching_losses(
        mosfet,
        request.operating_point["voltage"],
        request.operating_point["current"],
        request.operating_point["frequency"],
        request.temperature
    )
    
    conduction_loss = calculate_conduction_losses(
        mosfet,
        request.operating_point["current"],
        request.operating_point.get("duty_cycle", 0.5),
        request.temperature
    )
    
    return {
        "switching_loss": switching_loss,
        "conduction_loss": conduction_loss,
        "total_loss": switching_loss + conduction_loss,
        "efficiency": 1 - (switching_loss + conduction_loss) / request.operating_point.get("power", 1000)
    }


@app.post("/calc/thermal")
async def calculate_thermal(request: ThermalCalculationRequest):
    """Calculate junction temperature"""
    thermal = ThermalModel(
        rth_jc=request.device_rth_jc,
        rth_ch=request.heatsink_rth,
        cth_j=0.01,  # Thermal capacitance
        cth_h=1.0
    )
    
    tj = thermal.calculate_steady_state_temperature(
        request.power_loss,
        request.ambient_temp
    )
    
    # Calculate transient response
    time_constant = thermal.calculate_time_constant()
    
    return {
        "junction_temperature": tj,
        "case_temperature": tj - request.power_loss * request.device_rth_jc,
        "heatsink_temperature": request.ambient_temp + request.power_loss * request.heatsink_rth,
        "thermal_time_constant": time_constant,
        "thermal_margin": 150 - tj  # Assuming 150°C max junction temp
    }


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(registry), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)