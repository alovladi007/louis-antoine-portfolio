# IoT Robotics Manufacturing - Hybrid Implementation

A production-ready hybrid system that bridges the demo visualization with real backend services for IoT robotics in semiconductor manufacturing.

## 🚀 Quick Start

```bash
# Clone the repository
cd iot-robotics-backend

# Install dependencies
cd backend && npm install && cd ..

# Start all services with Docker
docker-compose up -d

# Start the Node.js backend
cd backend && npm run dev
```

The system will be available at:
- **Backend API**: http://localhost:4000
- **WebSocket**: ws://localhost:4001
- **ML Service**: http://localhost:8001
- **MQTT Broker**: mqtt://localhost:1883 (WebSocket on 9001)

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   HTML Demo     │────▶│   Node.js       │◀────│   ML Service    │
│  (Browser)      │ WS  │   Backend       │     │   (FastAPI)     │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                  │ MQTT
                        ┌─────────▼─────────┐
                        │   MQTT Broker     │
                        │   (Mosquitto)     │
                        └─────────┬─────────┘
                                  │
                        ┌─────────▼─────────┐
                        │   Python Edge     │
                        │     Agent         │
                        └───────────────────┘
```

## 📦 Components

### 1. **Node.js Backend** (`backend/`)
- REST API for robot status and commands
- WebSocket server for real-time updates
- MQTT client for telemetry ingestion
- In-memory storage (upgradeable to TimescaleDB)

### 2. **Python Edge Agent** (`edge-agents/`)
- Simulates realistic sensor data
- Generates vibration signals with fault frequencies
- Publishes telemetry via MQTT
- Handles robot commands

### 3. **ML Inference Service** (`ml-service/`)
- FastAPI service for RUL prediction
- Confidence interval calculation
- Health scoring and maintenance prioritization

### 4. **MQTT Broker** (Mosquitto)
- Message broker for telemetry
- WebSocket support for browser clients
- Persistent message storage

## 🔌 Connecting the HTML Demo

Update your `iot-robotics-project.html` to connect to the real backend:

```javascript
// Replace the simulated WebSocket with:
const ws = new WebSocket('ws://localhost:4001');

ws.onopen = () => {
    console.log('Connected to real backend');
    ws.send(JSON.stringify({ type: 'subscribe', channel: 'all' }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'telemetry':
            updateTelemetryChart(data.payload);
            break;
        case 'robot_update':
            updateRobotStatus(data.payload);
            break;
        case 'cleanroom':
            updateCleanroomMetrics(data.payload);
            break;
        case 'alert':
            showAlert(data.payload);
            break;
    }
};

// Fetch real data from API
async function fetchRobots() {
    const response = await fetch('http://localhost:4000/api/robots');
    const robots = await response.json();
    updateRobotDisplay(robots);
}

// Send commands
async function sendRobotCommand(robotId, command) {
    await fetch(`http://localhost:4000/api/robots/${robotId}/command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command, parameters: {} })
    });
}
```

## 📊 API Endpoints

### REST API (Port 4000)

- `GET /health` - System health check
- `GET /api/robots` - List all robots
- `GET /api/robots/:id` - Get robot details
- `POST /api/robots/:id/command` - Send robot command
- `GET /api/cleanroom/status` - Cleanroom metrics
- `GET /api/telemetry/latest` - Recent telemetry data
- `POST /api/ml/predict` - RUL prediction

### WebSocket Events (Port 4001)

**Incoming:**
- `init` - Initial state on connection
- `telemetry` - Real-time sensor data
- `robot_update` - Robot status changes
- `cleanroom` - Environmental updates
- `alert` - System alerts

**Outgoing:**
- `subscribe` - Subscribe to channels
- `command` - Send robot commands
- `ping` - Keep-alive

### ML Service (Port 8001)

- `GET /health` - Service health
- `POST /predict` - Single RUL prediction
- `POST /batch_predict` - Batch predictions
- `GET /model_info` - Model metadata

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root:

```env
# Backend
PORT=4000
WS_PORT=4001
MQTT_BROKER=localhost
NODE_ENV=development

# Edge Agent
SIMULATION_MODE=true
MQTT_PORT=1883

# ML Service
MODEL_PATH=/app/models
REDIS_URL=redis://localhost:6379
```

### MQTT Topics

- `factory/main/{robot_id}/telemetry` - Robot telemetry
- `factory/main/{robot_id}/cmd` - Robot commands
- `factory/main/{robot_id}/alerts` - Robot alerts
- `cleanroom/main/status` - Cleanroom data

## 🧪 Testing

### Test WebSocket Connection
```javascript
const ws = new WebSocket('ws://localhost:4001');
ws.onopen = () => ws.send(JSON.stringify({ type: 'ping' }));
ws.onmessage = (e) => console.log('Received:', JSON.parse(e.data));
```

### Test MQTT Publishing
```python
import paho.mqtt.client as mqtt
import json

client = mqtt.Client()
client.connect("localhost", 1883)
client.publish("factory/main/robot-001/telemetry", json.dumps({
    "temperature": 45.2,
    "vibration": {"rms": 0.35},
    "health_score": 85
}))
```

### Test ML Prediction
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "robot_id": "robot-001",
    "features": [0.35, 45.2, 6.5, 3.8, 4.2],
    "confidence_level": 0.95
  }'
```

## 📈 Monitoring

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f edge-agent

# Node.js backend
cd backend && npm run dev
```

### Check Service Health
```bash
# Backend
curl http://localhost:4000/health

# ML Service
curl http://localhost:8001/health

# MQTT
docker exec -it iot-robotics-backend_mosquitto_1 mosquitto_sub -t '#' -v
```

## 🚀 Production Deployment

### 1. Enable Database Persistence

Replace in-memory storage with TimescaleDB:

```javascript
// backend/db.js
const { Pool } = require('pg');
const pool = new Pool({
    connectionString: process.env.DATABASE_URL
});
```

### 2. Add Authentication

Implement JWT authentication:

```javascript
// backend/auth.js
const jwt = require('jsonwebtoken');
// ... authentication middleware
```

### 3. Enable TLS

Configure TLS certificates for MQTT and WebSocket:

```yaml
# docker-compose.yml
mosquitto:
  volumes:
    - ./certs:/mosquitto/certs:ro
```

### 4. Scale Services

Use Kubernetes for production:

```bash
kubectl apply -f k8s/
```

## 🛠️ Troubleshooting

### MQTT Connection Failed
```bash
# Check if Mosquitto is running
docker ps | grep mosquitto

# View Mosquitto logs
docker logs iot-robotics-backend_mosquitto_1
```

### WebSocket Not Connecting
```bash
# Check if backend is running
curl http://localhost:4000/health

# Check WebSocket port
netstat -an | grep 4001
```

### Edge Agent Not Publishing
```bash
# Check agent logs
docker logs iot-robotics-backend_edge-agent_1

# Test MQTT manually
mosquitto_sub -h localhost -t '#' -v
```

## 📚 Next Steps

1. **Integrate with HTML Demo**: Update the frontend to use real WebSocket/API
2. **Add Database**: Implement TimescaleDB for time-series storage
3. **Train ML Model**: Replace simulated model with trained LSTM/Random Forest
4. **Add Authentication**: Implement JWT-based auth
5. **Deploy to Cloud**: Use Docker Swarm or Kubernetes

## 📄 License

MIT

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.