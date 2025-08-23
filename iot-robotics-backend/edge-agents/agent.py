#!/usr/bin/env python3
"""
Edge Agent for IoT Robotics - Sensor Simulation and Telemetry
"""

import os
import json
import time
import random
import threading
from datetime import datetime
import numpy as np
from scipy import signal
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

class EdgeAgent:
    def __init__(self):
        self.mqtt_broker = os.getenv('MQTT_BROKER', 'localhost')
        self.mqtt_port = int(os.getenv('MQTT_PORT', 1883))
        self.simulation_mode = os.getenv('SIMULATION_MODE', 'true').lower() == 'true'
        
        self.client = mqtt.Client(client_id=f"edge-agent-{random.randint(1000, 9999)}")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        self.robots = ['robot-001', 'robot-002', 'robot-003']
        self.running = True
        self.robot_states = {}
        
        # Initialize robot states
        for robot_id in self.robots:
            self.robot_states[robot_id] = {
                'hours_running': 0,
                'cycles': 0,
                'base_vibration': 0.1 + random.random() * 0.05,
                'degradation_rate': random.uniform(0.0001, 0.0003)
            }
    
    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected to MQTT broker with result code {rc}")
        # Subscribe to command topics
        for robot_id in self.robots:
            client.subscribe(f"factory/main/{robot_id}/cmd")
    
    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            topic_parts = msg.topic.split('/')
            robot_id = topic_parts[2]
            
            print(f"Received command for {robot_id}: {payload.get('command')}")
            
            # Handle commands
            if payload.get('command') == 'emergency_stop':
                self.emergency_stop(robot_id)
            elif payload.get('command') == 'reset':
                self.reset_robot(robot_id)
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def connect(self):
        """Connect to MQTT broker"""
        try:
            self.client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.client.loop_start()
            print(f"Connecting to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}")
            time.sleep(2)  # Give time to connect
            return True
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def generate_vibration_signal(self, robot_id):
        """Generate realistic vibration signal with bearing fault frequencies"""
        state = self.robot_states[robot_id]
        
        # Base parameters
        fs = 10000  # Sampling frequency
        duration = 0.1  # 100ms window
        t = np.linspace(0, duration, int(fs * duration))
        
        # Degradation factor based on running hours
        degradation = state['hours_running'] * state['degradation_rate']
        
        # Base vibration (increases with degradation)
        base_amplitude = state['base_vibration'] * (1 + degradation)
        signal_base = np.random.normal(0, base_amplitude, len(t))
        
        # Add bearing fault frequencies (if degraded)
        if degradation > 0.1:
            # BPFO (Ball Pass Frequency Outer race) ~3.5x shaft speed
            bpfo = 105 * (1 + random.uniform(-0.1, 0.1))
            fault_amplitude = degradation * 0.5
            signal_base += fault_amplitude * np.sin(2 * np.pi * bpfo * t)
            
            # Add harmonics
            signal_base += fault_amplitude * 0.3 * np.sin(2 * np.pi * 2 * bpfo * t)
        
        # Calculate features
        rms = np.sqrt(np.mean(signal_base**2))
        peak = np.max(np.abs(signal_base))
        crest_factor = peak / rms if rms > 0 else 0
        kurtosis = np.mean((signal_base - np.mean(signal_base))**4) / (np.std(signal_base)**4) if np.std(signal_base) > 0 else 3
        
        return {
            'rms': float(rms),
            'peak': float(peak),
            'crest_factor': float(crest_factor),
            'kurtosis': float(kurtosis)
        }
    
    def generate_telemetry(self, robot_id):
        """Generate telemetry data for a robot"""
        state = self.robot_states[robot_id]
        
        # Update running hours
        state['hours_running'] += 0.001  # Increment by ~3.6 seconds
        state['cycles'] += 1
        
        # Generate vibration data
        vibration = self.generate_vibration_signal(robot_id)
        
        # Temperature (increases with running time and vibration)
        base_temp = 30
        temp_rise = state['hours_running'] * 0.01 + vibration['rms'] * 50
        temperature = base_temp + temp_rise + random.gauss(0, 2)
        
        # Current (correlates with load and degradation)
        base_current = 5.0
        current = base_current + vibration['rms'] * 10 + random.gauss(0, 0.5)
        
        # Health score (decreases with degradation)
        degradation = state['hours_running'] * state['degradation_rate']
        health_score = max(0, 100 * (1 - degradation))
        
        # RUL prediction (simplified)
        rul_hours = max(0, 2000 - state['hours_running'])
        
        # Determine status
        if health_score < 20:
            status = 'maintenance'
        elif random.random() > 0.9:
            status = 'idle'
        else:
            status = 'running'
        
        telemetry = {
            'robot_id': robot_id,
            'timestamp': datetime.now().isoformat(),
            'temperature': float(temperature),
            'vibration': vibration,
            'current': float(current),
            'health_score': float(health_score),
            'rul_hours': float(rul_hours),
            'status': status,
            'cycles': state['cycles'],
            'running_hours': state['hours_running']
        }
        
        return telemetry
    
    def generate_cleanroom_data(self):
        """Generate cleanroom environmental data"""
        # Simulate daily patterns
        hour = datetime.now().hour
        
        # Particle count (lower during night shifts)
        base_particles = 200 if 8 <= hour <= 18 else 150
        particles = base_particles + random.gauss(0, 30)
        
        # Occasional excursions
        if random.random() < 0.01:
            particles += random.uniform(200, 1000)
        
        # Airflow (relatively stable)
        airflow = 0.5 + random.gauss(0, 0.02)
        
        # Humidity (slight daily variation)
        humidity = 45 + 3 * np.sin(hour * np.pi / 12) + random.gauss(0, 1)
        
        # Temperature (controlled)
        temperature = 21.5 + 0.5 * np.sin(hour * np.pi / 12) + random.gauss(0, 0.3)
        
        return {
            'particles': float(max(0, particles)),
            'airflow': float(max(0.3, min(0.7, airflow))),
            'humidity': float(max(30, min(60, humidity))),
            'temperature': float(max(19, min(24, temperature))),
            'timestamp': datetime.now().isoformat()
        }
    
    def check_alerts(self, robot_id, telemetry):
        """Check for alert conditions"""
        alerts = []
        
        # Vibration alert
        if telemetry['vibration']['rms'] > 0.5:
            alerts.append({
                'type': 'vibration_high',
                'severity': 'critical' if telemetry['vibration']['rms'] > 0.7 else 'warning',
                'message': f"High vibration detected: {telemetry['vibration']['rms']:.3f} g RMS",
                'value': telemetry['vibration']['rms']
            })
        
        # Temperature alert
        if telemetry['temperature'] > 70:
            alerts.append({
                'type': 'temperature_high',
                'severity': 'critical' if telemetry['temperature'] > 80 else 'warning',
                'message': f"High temperature: {telemetry['temperature']:.1f}°C",
                'value': telemetry['temperature']
            })
        
        # Health alert
        if telemetry['health_score'] < 30:
            alerts.append({
                'type': 'health_low',
                'severity': 'critical' if telemetry['health_score'] < 20 else 'warning',
                'message': f"Low health score: {telemetry['health_score']:.1f}%",
                'value': telemetry['health_score']
            })
        
        # RUL alert
        if telemetry['rul_hours'] < 100:
            alerts.append({
                'type': 'maintenance_required',
                'severity': 'critical' if telemetry['rul_hours'] < 24 else 'warning',
                'message': f"Maintenance required in {telemetry['rul_hours']:.0f} hours",
                'value': telemetry['rul_hours']
            })
        
        return alerts
    
    def publish_telemetry(self):
        """Main telemetry publishing loop"""
        while self.running:
            try:
                # Publish robot telemetry
                for robot_id in self.robots:
                    telemetry = self.generate_telemetry(robot_id)
                    topic = f"factory/main/{robot_id}/telemetry"
                    
                    self.client.publish(topic, json.dumps(telemetry), qos=1)
                    print(f"Published telemetry for {robot_id}: health={telemetry['health_score']:.1f}%, RUL={telemetry['rul_hours']:.0f}h")
                    
                    # Check for alerts
                    alerts = self.check_alerts(robot_id, telemetry)
                    for alert in alerts:
                        alert['robot_id'] = robot_id
                        alert['timestamp'] = datetime.now().isoformat()
                        alert_topic = f"factory/main/{robot_id}/alerts"
                        self.client.publish(alert_topic, json.dumps(alert), qos=2)
                        print(f"Alert for {robot_id}: {alert['type']} ({alert['severity']})")
                
                # Publish cleanroom data
                cleanroom_data = self.generate_cleanroom_data()
                self.client.publish("cleanroom/main/status", json.dumps(cleanroom_data), qos=1)
                print(f"Published cleanroom data: particles={cleanroom_data['particles']:.0f}/m³")
                
                # Sleep interval
                time.sleep(2)  # Publish every 2 seconds
                
            except Exception as e:
                print(f"Error in telemetry loop: {e}")
                time.sleep(5)
    
    def emergency_stop(self, robot_id):
        """Handle emergency stop command"""
        print(f"EMERGENCY STOP for {robot_id}")
        # Reset state
        self.robot_states[robot_id]['base_vibration'] = 0.05
        # Send confirmation
        self.client.publish(
            f"factory/main/{robot_id}/status",
            json.dumps({
                'status': 'emergency_stopped',
                'timestamp': datetime.now().isoformat()
            }),
            qos=2
        )
    
    def reset_robot(self, robot_id):
        """Reset robot state"""
        print(f"Resetting {robot_id}")
        self.robot_states[robot_id] = {
            'hours_running': 0,
            'cycles': 0,
            'base_vibration': 0.1 + random.random() * 0.05,
            'degradation_rate': random.uniform(0.0001, 0.0003)
        }
    
    def run(self):
        """Main run method"""
        print("Starting IoT Robotics Edge Agent...")
        
        # Connect to MQTT
        if not self.connect():
            print("Failed to connect to MQTT broker. Retrying in 10 seconds...")
            time.sleep(10)
            return self.run()
        
        print("Edge Agent running in simulation mode" if self.simulation_mode else "Edge Agent running in production mode")
        
        # Start telemetry thread
        telemetry_thread = threading.Thread(target=self.publish_telemetry)
        telemetry_thread.daemon = True
        telemetry_thread.start()
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down edge agent...")
            self.running = False
            self.client.loop_stop()
            self.client.disconnect()

if __name__ == "__main__":
    agent = EdgeAgent()
    agent.run()