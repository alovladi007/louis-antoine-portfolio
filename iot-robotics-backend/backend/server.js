const express = require('express');
const WebSocket = require('ws');
const mqtt = require('mqtt');
const cors = require('cors');
const { v4: uuidv4 } = require('uuid');
const RobotController = require('./robot-controller');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 4000;
const WS_PORT = process.env.WS_PORT || 4001;

// Middleware
app.use(cors());
app.use(express.json());

// In-memory storage for demo (replace with database in production)
const telemetryBuffer = [];
const robotStatus = new Map();
const robotControllers = new Map();
const cleanroomData = {
  particles: 152,
  airflow: 0.52,
  humidity: 45,
  temperature: 21.5
};

// Initialize robot status and controllers
['robot-001', 'robot-002', 'robot-003'].forEach(id => {
  // Create robot controller
  const controller = new RobotController(id);
  robotControllers.set(id, controller);
  
  // Initialize status
  robotStatus.set(id, {
    id,
    name: `Robot ${id.split('-')[1]}`,
    status: 'idle',
    health: 100,
    rul: 2000,
    temperature: 30,
    vibration: 0.1,
    current: 5.0,
    controller: controller.getState(),
    lastUpdate: new Date()
  });
});

// MQTT Client
const mqttClient = mqtt.connect(`mqtt://${process.env.MQTT_BROKER || 'localhost'}:1883`);

mqttClient.on('connect', () => {
  console.log('Connected to MQTT broker');
  
  // Subscribe to telemetry topics
  mqttClient.subscribe('factory/+/+/telemetry');
  mqttClient.subscribe('factory/+/+/alerts');
  mqttClient.subscribe('cleanroom/+/status');
});

mqttClient.on('message', (topic, message) => {
  try {
    const data = JSON.parse(message.toString());
    const topicParts = topic.split('/');
    
    if (topic.includes('telemetry')) {
      handleTelemetry(topicParts[2], data);
    } else if (topic.includes('alerts')) {
      handleAlert(topicParts[2], data);
    } else if (topic.includes('cleanroom')) {
      handleCleanroomUpdate(data);
    }
  } catch (error) {
    console.error('Error processing MQTT message:', error);
  }
});

// WebSocket Server
const wss = new WebSocket.Server({ port: WS_PORT });
const wsClients = new Set();

wss.on('connection', (ws) => {
  const clientId = uuidv4();
  ws.clientId = clientId;
  wsClients.add(ws);
  
  console.log(`WebSocket client connected: ${clientId}`);
  
  // Send initial state
  ws.send(JSON.stringify({
    type: 'init',
    payload: {
      robots: Array.from(robotStatus.values()),
      cleanroom: cleanroomData
    }
  }));
  
  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);
      handleWebSocketMessage(ws, data);
    } catch (error) {
      console.error('WebSocket message error:', error);
    }
  });
  
  ws.on('close', () => {
    wsClients.delete(ws);
    console.log(`WebSocket client disconnected: ${clientId}`);
  });
  
  ws.on('error', (error) => {
    console.error(`WebSocket error for ${clientId}:`, error);
  });
});

// Broadcast to all WebSocket clients
function broadcast(data) {
  const message = JSON.stringify(data);
  wsClients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  });
}

// Handle telemetry data
function handleTelemetry(deviceId, data) {
  // Update robot status
  if (robotStatus.has(deviceId)) {
    const robot = robotStatus.get(deviceId);
    robot.temperature = data.temperature || robot.temperature;
    robot.vibration = data.vibration?.rms || robot.vibration;
    robot.current = data.current || robot.current;
    robot.health = data.health_score || robot.health;
    robot.rul = data.rul_hours || robot.rul;
    robot.lastUpdate = new Date();
    robot.status = data.status || 'running';
    
    // Broadcast update
    broadcast({
      type: 'robot_update',
      payload: robot
    });
  }
  
  // Buffer telemetry
  telemetryBuffer.push({
    deviceId,
    timestamp: new Date(),
    data
  });
  
  // Keep buffer size manageable
  if (telemetryBuffer.length > 1000) {
    telemetryBuffer.shift();
  }
  
  // Broadcast telemetry
  broadcast({
    type: 'telemetry',
    payload: {
      deviceId,
      timestamp: new Date(),
      value: data.vibration?.rms || Math.random() * 0.5,
      metric: 'vibration'
    }
  });
}

// Handle alerts
function handleAlert(deviceId, data) {
  console.log(`Alert for ${deviceId}:`, data);
  
  broadcast({
    type: 'alert',
    payload: {
      deviceId,
      ...data
    }
  });
}

// Handle cleanroom updates
function handleCleanroomUpdate(data) {
  Object.assign(cleanroomData, data);
  
  broadcast({
    type: 'cleanroom',
    payload: cleanroomData
  });
}

// Handle WebSocket messages from clients
async function handleWebSocketMessage(ws, data) {
  switch (data.type) {
    case 'subscribe':
      console.log(`Client ${ws.clientId} subscribed to ${data.channel}`);
      if (data.channel === 'robot_control') {
        // Send current robot states
        const states = {};
        robotControllers.forEach((controller, id) => {
          states[id] = controller.getState();
        });
        ws.send(JSON.stringify({
          type: 'robot_states',
          payload: states
        }));
      }
      break;
      
    case 'robot_jog':
      const jogController = robotControllers.get(data.robotId);
      if (jogController) {
        const result = jogController.jog(data.axis, data.direction, data.speed);
        broadcast({
          type: 'robot_state_update',
          payload: {
            robotId: data.robotId,
            state: jogController.getState(),
            jogResult: result
          }
        });
      }
      break;
      
    case 'robot_move':
      const moveController = robotControllers.get(data.robotId);
      if (moveController) {
        try {
          const result = await moveController.moveTo(data.target, data.options);
          broadcast({
            type: 'robot_motion_complete',
            payload: {
              robotId: data.robotId,
              state: moveController.getState(),
              result
            }
          });
        } catch (error) {
          ws.send(JSON.stringify({
            type: 'error',
            payload: { robotId: data.robotId, error: error.message }
          }));
        }
      }
      break;
      
    case 'robot_stop':
      const stopController = robotControllers.get(data.robotId);
      if (stopController) {
        stopController.stop();
        broadcast({
          type: 'emergency_stop',
          payload: {
            robotId: data.robotId,
            state: stopController.getState()
          }
        });
      }
      break;
      
    case 'robot_home':
      const homeController = robotControllers.get(data.robotId);
      if (homeController) {
        try {
          const result = await homeController.home();
          broadcast({
            type: 'robot_home_complete',
            payload: {
              robotId: data.robotId,
              state: homeController.getState(),
              result
            }
          });
        } catch (error) {
          ws.send(JSON.stringify({
            type: 'error',
            payload: { robotId: data.robotId, error: error.message }
          }));
        }
      }
      break;
      
    case 'command':
      // Forward command to MQTT
      const topic = `factory/main/${data.robotId}/cmd`;
      mqttClient.publish(topic, JSON.stringify(data.payload));
      break;
      
    case 'ping':
      ws.send(JSON.stringify({ type: 'pong' }));
      break;
  }
}

// REST API Endpoints
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    mqtt: mqttClient.connected,
    wsClients: wsClients.size,
    timestamp: new Date()
  });
});

app.get('/api/robots', (req, res) => {
  const robots = Array.from(robotStatus.values()).map(robot => ({
    ...robot,
    controller: robotControllers.get(robot.id).getState()
  }));
  res.json(robots);
});

app.get('/api/robots/:id', (req, res) => {
  const robot = robotStatus.get(req.params.id);
  const controller = robotControllers.get(req.params.id);
  
  if (robot && controller) {
    res.json({
      ...robot,
      controller: controller.getState()
    });
  } else {
    res.status(404).json({ error: 'Robot not found' });
  }
});

app.get('/api/robots/:id/state', (req, res) => {
  const controller = robotControllers.get(req.params.id);
  
  if (controller) {
    res.json(controller.getState());
  } else {
    res.status(404).json({ error: 'Robot not found' });
  }
});

app.post('/api/robots/:id/move', async (req, res) => {
  const controller = robotControllers.get(req.params.id);
  
  if (!controller) {
    return res.status(404).json({ error: 'Robot not found' });
  }
  
  try {
    const result = await controller.moveTo(req.body.target, req.body.options);
    
    // Broadcast state update
    broadcast({
      type: 'robot_motion',
      payload: {
        robotId: req.params.id,
        state: controller.getState(),
        result
      }
    });
    
    res.json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.post('/api/robots/:id/jog', (req, res) => {
  const controller = robotControllers.get(req.params.id);
  
  if (!controller) {
    return res.status(404).json({ error: 'Robot not found' });
  }
  
  const { axis, direction, speed } = req.body;
  const result = controller.jog(axis, direction, speed);
  
  // Broadcast state update
  broadcast({
    type: 'robot_jog',
    payload: {
      robotId: req.params.id,
      state: controller.getState()
    }
  });
  
  res.json(result);
});

app.post('/api/robots/:id/home', async (req, res) => {
  const controller = robotControllers.get(req.params.id);
  
  if (!controller) {
    return res.status(404).json({ error: 'Robot not found' });
  }
  
  try {
    const result = await controller.home();
    
    // Broadcast state update
    broadcast({
      type: 'robot_home',
      payload: {
        robotId: req.params.id,
        state: controller.getState(),
        result
      }
    });
    
    res.json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.post('/api/robots/:id/stop', (req, res) => {
  const controller = robotControllers.get(req.params.id);
  
  if (!controller) {
    return res.status(404).json({ error: 'Robot not found' });
  }
  
  const result = controller.stop();
  
  // Broadcast emergency stop
  broadcast({
    type: 'emergency_stop',
    payload: {
      robotId: req.params.id,
      state: controller.getState()
    }
  });
  
  res.json(result);
});

app.post('/api/robots/:id/reset', (req, res) => {
  const controller = robotControllers.get(req.params.id);
  
  if (!controller) {
    return res.status(404).json({ error: 'Robot not found' });
  }
  
  const result = controller.reset();
  
  // Broadcast reset
  broadcast({
    type: 'robot_reset',
    payload: {
      robotId: req.params.id,
      state: controller.getState()
    }
  });
  
  res.json(result);
});

app.post('/api/robots/:id/program', async (req, res) => {
  const controller = robotControllers.get(req.params.id);
  
  if (!controller) {
    return res.status(404).json({ error: 'Robot not found' });
  }
  
  try {
    const results = await controller.executeProgram(req.body.program);
    
    // Broadcast completion
    broadcast({
      type: 'program_complete',
      payload: {
        robotId: req.params.id,
        state: controller.getState(),
        results
      }
    });
    
    res.json({ success: true, results });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.post('/api/robots/:id/command', (req, res) => {
  const { command, parameters } = req.body;
  const topic = `factory/main/${req.params.id}/cmd`;
  
  mqttClient.publish(topic, JSON.stringify({
    command,
    parameters,
    timestamp: new Date()
  }));
  
  res.json({ success: true, message: 'Command sent' });
});

app.get('/api/cleanroom/status', (req, res) => {
  res.json(cleanroomData);
});

app.get('/api/telemetry/latest', (req, res) => {
  const latest = telemetryBuffer.slice(-100);
  res.json(latest);
});

app.post('/api/ml/predict', async (req, res) => {
  try {
    // Forward to ML service
    const response = await fetch('http://localhost:8001/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    
    const result = await response.json();
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: 'ML service unavailable' });
  }
});

// Broadcast robot states periodically
function broadcastRobotStates() {
  setInterval(() => {
    const states = {};
    robotControllers.forEach((controller, id) => {
      const state = controller.getState();
      states[id] = state;
      
      // Update robot status
      const robot = robotStatus.get(id);
      if (robot) {
        robot.controller = state;
        robot.lastUpdate = new Date();
      }
    });
    
    broadcast({
      type: 'robot_states_update',
      payload: states
    });
  }, 100); // 10Hz update rate
}

// Simulate data generation (remove in production)
function simulateData() {
  setInterval(() => {
    // Simulate robot telemetry
    ['robot-001', 'robot-002', 'robot-003'].forEach(id => {
      const robot = robotStatus.get(id);
      if (robot && Math.random() > 0.3) {
        const telemetry = {
          temperature: 30 + Math.random() * 20,
          vibration: {
            rms: 0.1 + Math.random() * 0.4,
            peak: 0.3 + Math.random() * 0.5,
            crest_factor: 3 + Math.random()
          },
          current: 5 + Math.random() * 3,
          health_score: Math.max(0, robot.health - Math.random() * 0.1),
          rul_hours: Math.max(0, robot.rul - 0.1),
          status: Math.random() > 0.9 ? 'idle' : 'running'
        };
        
        handleTelemetry(id, telemetry);
      }
    });
    
    // Simulate cleanroom data
    const cleanroomUpdate = {
      particles: 150 + Math.random() * 100,
      airflow: 0.45 + Math.random() * 0.15,
      humidity: 40 + Math.random() * 10,
      temperature: 20 + Math.random() * 3
    };
    
    handleCleanroomUpdate(cleanroomUpdate);
    
    // Occasional alerts
    if (Math.random() > 0.95) {
      const robotId = ['robot-001', 'robot-002', 'robot-003'][Math.floor(Math.random() * 3)];
      handleAlert(robotId, {
        type: 'vibration_warning',
        severity: 'warning',
        message: 'Vibration exceeds threshold',
        value: 0.8 + Math.random() * 0.2
      });
    }
  }, 2000); // Update every 2 seconds
}

// Start simulation in development mode
if (process.env.NODE_ENV !== 'production') {
  simulateData();
  broadcastRobotStates();
}

// Start HTTP server
app.listen(PORT, () => {
  console.log(`HTTP server running on port ${PORT}`);
  console.log(`WebSocket server running on port ${WS_PORT}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, closing connections...');
  mqttClient.end();
  wss.close();
  process.exit(0);
});