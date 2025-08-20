# Advanced Power Electronics Platform

## ⚡ High Voltage Safety Warning

**⚠️ DANGER: HIGH VOLTAGE**

This platform interfaces with high-voltage power electronics systems. Improper use can result in:
- Electric shock or electrocution
- Fire hazard
- Equipment damage
- Personal injury or death

**Safety Requirements:**
- Only qualified personnel should operate HIL systems
- Always verify proper grounding and isolation
- Use appropriate PPE (insulated gloves, safety glasses)
- Implement hardware interlocks and E-STOP circuits
- Never work on energized circuits
- Follow lockout/tagout procedures

## 🚀 Overview

Advanced Power Electronics Platform for EV and Renewable Energy applications, featuring:
- Real-time power electronics simulation (DC-DC, inverters, motor control)
- Hardware-in-the-Loop (HIL) testing with STM32 control
- Machine learning for predictive maintenance and optimization
- Complete telemetry pipeline with TimescaleDB
- Web-based design studio and analytics

### Key Features

- **Topologies**: Interleaved Boost, LLC Resonant, Bidirectional DC-DC, 3-Phase Inverter
- **Control**: FOC motor control, MPPT algorithms, Grid-tie PLL
- **Analysis**: Loss calculation, thermal modeling, THD analysis
- **HIL**: CAN-based real hardware control with safety interlocks
- **ML**: Predictive maintenance, anomaly detection, MPC optimization

## 📋 Prerequisites

- Node.js 18+ and pnpm 8+
- Python 3.11+
- Docker and Docker Compose
- (For HIL) Raspberry Pi + STM32G4 + CAN interface

## 🏃 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/your-org/power-electronics.git
cd power-electronics
pnpm install
```

### 2. Environment Setup

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Start Services

```bash
# Start all services with Docker
docker compose up --build

# Or run services individually for development
pnpm dev
```

### 4. Access Applications

- **Web App**: http://localhost:3000 (admin@demo.local / Demo123!)
- **API Docs**: http://localhost:4000/api
- **Simulation Service**: http://localhost:5000/docs
- **ML Service**: http://localhost:5001/docs
- **Grafana**: http://localhost:3001 (admin/admin)

## 🏗️ Architecture

```
power-electronics/
├── apps/
│   ├── web/          # Next.js 14 frontend
│   ├── api/          # NestJS API gateway
│   └── docs/         # Docusaurus documentation
├── services/
│   ├── sim/          # Python simulation service
│   ├── ml/           # Python ML service
│   ├── stream/       # MQTT gateway
│   └── hil-agent/    # Raspberry Pi HIL agent
├── firmware/
│   └── control-fw/   # STM32G4 control firmware
├── packages/
│   └── shared/       # Shared schemas and types
├── infra/
│   ├── docker-compose.yml
│   └── init-sql/     # TimescaleDB setup
└── shared/
    └── can/          # CAN DBC files
```

## 🔬 Simulation Capabilities

### Supported Topologies

1. **Interleaved Boost Converter**
   - 2-4 phase operation
   - Current ripple cancellation
   - Phase shedding for light load

2. **LLC Resonant Converter**
   - Zero-voltage switching (ZVS)
   - Frequency control
   - Gain curves

3. **Bidirectional DC-DC**
   - Buck/Boost operation
   - Synchronous rectification
   - EV OBC and traction applications

4. **3-Phase Inverter**
   - SVPWM/DPWM modulation
   - FOC motor control
   - Grid-tie with PLL

### Control Algorithms

- **Motor Control**: Field-Oriented Control (FOC) with Clarke/Park transforms
- **MPPT**: Perturb & Observe, Incremental Conductance
- **Grid-Tie**: SRF-PLL, dq current control
- **DC-DC**: Current-mode control, resonant tank control

## 🔧 Hardware-in-the-Loop (HIL)

### Hardware Setup

```
┌─────────────┐     CAN      ┌──────────────┐     PWM/ADC    ┌──────────────┐
│ Raspberry   │◄────────────►│   STM32G4    │◄─────────────►│ Power Stage  │
│   Pi 5      │              │   Control    │               │  + Drivers   │
└─────────────┘              └──────────────┘               └──────────────┘
      │                             │                              │
      │ MQTT                        │ Debug                       │ HV
      ▼                             ▼                              ▼
┌─────────────┐              ┌──────────────┐               ┌──────────────┐
│   Cloud     │              │   J-Link     │               │   DC Link    │
│  Services   │              │   Debugger   │               │   Supply     │
└─────────────┘              └──────────────┘               └──────────────┘
```

### CAN Protocol

See `shared/can/bench.dbc` for complete message definitions:
- Commands: 0x100-0x102 (control, setpoints)
- Telemetry: 0x200-0x203 (pack, phase, motor, thermal)
- Faults: 0x220 (protection status)

### Safety Interlocks

- Hardware E-STOP with latching relay
- Overcurrent protection (hardware comparator)
- Desat detection on gate drivers
- Software limits in firmware
- Pre-charge sequence enforcement

## 🤖 Machine Learning

### Predictive Maintenance
- LSTM/TCN models for failure prediction
- Anomaly detection on THD and temperature patterns
- Remaining useful life estimation

### Control Optimization
- Model Predictive Control (MPC) for ripple minimization
- Reinforcement Learning (SAC) for efficiency optimization
- Real-time inference with ONNX runtime

## 📊 Data Pipeline

```
Telemetry → MQTT → TimescaleDB → Grafana
    ↓         ↓          ↓           ↓
  WebSocket  Redis    Analytics   Dashboards
```

### Database Schema

- **Projects**: Design configurations and metadata
- **Runs**: Simulation and HIL test executions
- **Telemetry**: Time-series data (hypertable)
- **Components**: Device library (MOSFETs, magnetics)

## 🧪 Testing

```bash
# Run all tests
pnpm test

# Unit tests only
pnpm test:unit

# E2E tests
pnpm test:e2e

# Python service tests
cd services/sim && pytest
cd services/ml && pytest

# Firmware tests (requires hardware)
cd firmware/control-fw && make test
```

## 📚 Documentation

- **Architecture Guide**: [docs/architecture.md](docs/architecture.md)
- **API Reference**: http://localhost:4000/api
- **HIL Setup**: [docs/hil.md](docs/hil.md)
- **Safety Procedures**: [docs/safety.md](docs/safety.md)

## 🚀 Deployment

### Development

```bash
docker compose up --build
```

### Production

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Kubernetes

```bash
kubectl apply -k k8s/overlays/production
```

## 🔐 Security

- JWT authentication with role-based access control
- Multi-tenant isolation via org_id
- MQTT mTLS for device authentication
- Audit logging for all commands
- Field-level encryption for sensitive data

## 📈 Performance

- Simulation: Up to 1MHz sample rate
- HIL: 10kHz control loop
- Telemetry: 100k samples/sec ingestion
- ML Inference: <10ms latency

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## ⚠️ Disclaimer

This software is provided for educational and research purposes. Users are responsible for:
- Ensuring safe operation of power electronics systems
- Compliance with local electrical codes and regulations
- Proper training and certification for high-voltage work
- Implementation of appropriate safety measures

The authors assume no liability for damages or injuries resulting from the use of this software.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/power-electronics/issues)
- **Documentation**: [https://docs.powerelec.io](https://docs.powerelec.io)
- **Community**: [Discord Server](https://discord.gg/powerelec)

---

**Remember: Safety First! Always de-energize circuits before working on them.**