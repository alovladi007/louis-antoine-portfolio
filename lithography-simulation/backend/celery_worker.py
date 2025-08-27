from celery import Celery
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import SimulationModel
from simulation_engine import LithographySimulator, OpticalMetrology, OPCProcessor
import redis
import pickle
import json
import os

# Initialize Celery
celery_app = Celery(
    'lithography_tasks',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3000,  # 50 minutes
)

# Initialize Redis client
redis_client = redis.Redis(host='redis', port=6379, decode_responses=False)

# Initialize simulators
litho_simulator = LithographySimulator()
metrology = OpticalMetrology()
opc_processor = OPCProcessor()

@celery_app.task(bind=True)
def run_simulation_task(self, simulation_id: int, simulation_type: str, parameters: dict):
    """Run simulation in background"""
    
    # Get database session
    db = SessionLocal()
    
    try:
        # Update simulation status
        simulation = db.query(SimulationModel).filter(SimulationModel.id == simulation_id).first()
        if not simulation:
            return {'error': 'Simulation not found'}
        
        simulation.status = 'running'
        db.commit()
        
        # Update progress in Redis
        redis_client.set(f"sim:{simulation_id}:status", "running")
        redis_client.set(f"sim:{simulation_id}:progress", "0")
        
        # Run appropriate simulation
        if simulation_type == 'lithography':
            results = litho_simulator.run_simulation(parameters)
            
        elif simulation_type == 'metrology':
            # Simplified metrology simulation
            results = {
                'measurements': {
                    'thickness': 100.5,
                    'roughness': 0.8,
                    'uniformity': 98.5
                },
                'uncertainty': 0.1
            }
            
        elif simulation_type == 'opc':
            # Simplified OPC simulation
            results = {
                'epe_improvement': 35.2,
                'ler_improvement': 28.7,
                'corner_improvement': 42.1
            }
            
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")
        
        # Store results
        redis_client.setex(
            f"sim:{simulation_id}:results",
            3600,  # 1 hour TTL
            pickle.dumps(results)
        )
        
        # Update database
        simulation.status = 'completed'
        simulation.results = results
        db.commit()
        
        # Update Redis status
        redis_client.set(f"sim:{simulation_id}:status", "completed")
        redis_client.set(f"sim:{simulation_id}:progress", "100")
        
        return {
            'status': 'completed',
            'simulation_id': simulation_id,
            'results_summary': {
                'type': simulation_type,
                'completed': True
            }
        }
        
    except Exception as e:
        # Update error status
        if simulation:
            simulation.status = 'failed'
            simulation.error_message = str(e)
            db.commit()
        
        redis_client.set(f"sim:{simulation_id}:status", "failed")
        
        raise
        
    finally:
        db.close()

@celery_app.task
def process_batch_simulations(simulation_ids: list, parameters: dict):
    """Process multiple simulations in batch"""
    results = []
    
    for sim_id in simulation_ids:
        try:
            result = run_simulation_task.delay(sim_id, 'lithography', parameters)
            results.append({
                'simulation_id': sim_id,
                'task_id': result.id,
                'status': 'queued'
            })
        except Exception as e:
            results.append({
                'simulation_id': sim_id,
                'error': str(e),
                'status': 'failed'
            })
    
    return results

@celery_app.task
def generate_report(simulation_ids: list, format: str = 'pdf'):
    """Generate report for simulations"""
    # Simplified report generation
    report_data = {
        'simulations': simulation_ids,
        'format': format,
        'generated_at': datetime.now().isoformat()
    }
    
    # Store report in Redis
    report_id = f"report:{uuid.uuid4().hex}"
    redis_client.setex(report_id, 3600, json.dumps(report_data))
    
    return {
        'report_id': report_id,
        'status': 'generated'
    }

if __name__ == '__main__':
    celery_app.start()