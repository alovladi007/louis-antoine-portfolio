-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create database if not exists
SELECT 'CREATE DATABASE powerelec'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'powerelec')\gexec

\c powerelec;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Organizations table
CREATE TABLE IF NOT EXISTS orgs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID REFERENCES orgs(id) ON DELETE CASCADE,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    role VARCHAR(50) NOT NULL CHECK (role IN ('Admin', 'Engineer', 'Viewer')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Projects table
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID REFERENCES orgs(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(50) NOT NULL CHECK (domain IN ('EV', 'PV', 'GRID', 'STORAGE')),
    description TEXT,
    metadata JSONB DEFAULT '{}',
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Designs table
CREATE TABLE IF NOT EXISTS designs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    topology VARCHAR(100) NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}',
    controller JSONB NOT NULL DEFAULT '{}',
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Components library
CREATE TABLE IF NOT EXISTS components (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID REFERENCES orgs(id) ON DELETE CASCADE,
    kind VARCHAR(50) NOT NULL CHECK (kind IN ('MOSFET', 'IGBT', 'DIODE', 'INDUCTOR', 'CAPACITOR', 'TRANSFORMER', 'HEATSINK')),
    vendor VARCHAR(100),
    part_number VARCHAR(100),
    params JSONB NOT NULL DEFAULT '{}',
    datasheet_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(vendor, part_number)
);

-- Simulation/HIL runs
CREATE TABLE IF NOT EXISTS runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    design_id UUID REFERENCES designs(id) ON DELETE CASCADE,
    mode VARCHAR(20) NOT NULL CHECK (mode IN ('SIM', 'HIL')),
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    summary JSONB DEFAULT '{}',
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Telemetry time-series data (hypertable)
CREATE TABLE IF NOT EXISTS telemetry (
    time TIMESTAMPTZ NOT NULL,
    run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    signal VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    metadata JSONB DEFAULT '{}'
);

-- Create hypertable for telemetry
SELECT create_hypertable('telemetry', 'time', 
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Create index on telemetry
CREATE INDEX IF NOT EXISTS idx_telemetry_run_signal ON telemetry (run_id, signal, time DESC);

-- HIL-specific tables
CREATE TABLE IF NOT EXISTS bench_devices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID REFERENCES orgs(id) ON DELETE CASCADE,
    device_id VARCHAR(100) UNIQUE NOT NULL,
    label VARCHAR(255) NOT NULL,
    can_profile VARCHAR(50),
    interface_name VARCHAR(50),
    last_seen_at TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'OFFLINE',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- CAN telemetry (hypertable)
CREATE TABLE IF NOT EXISTS bench_can (
    time TIMESTAMPTZ NOT NULL,
    device_id VARCHAR(100) NOT NULL,
    can_id INTEGER NOT NULL,
    dlc INTEGER NOT NULL,
    data BYTEA NOT NULL,
    ts_hw BIGINT
);

-- Create hypertable for CAN data
SELECT create_hypertable('bench_can', 'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Decoded signals from CAN (hypertable)
CREATE TABLE IF NOT EXISTS bench_signals (
    time TIMESTAMPTZ NOT NULL,
    device_id VARCHAR(100) NOT NULL,
    signal VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL
);

-- Create hypertable for bench signals
SELECT create_hypertable('bench_signals', 'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- ML models registry
CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID REFERENCES orgs(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL CHECK (type IN ('ANOMALY', 'PREDICTION', 'OPTIMIZATION')),
    version VARCHAR(50) NOT NULL,
    path TEXT,
    metrics JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(org_id, name, version)
);

-- Audit log
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_users_org ON users(org_id);
CREATE INDEX IF NOT EXISTS idx_projects_org ON projects(org_id);
CREATE INDEX IF NOT EXISTS idx_designs_project ON designs(project_id);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_components_kind ON components(kind);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at DESC);

-- Create continuous aggregates for telemetry
CREATE MATERIALIZED VIEW IF NOT EXISTS telemetry_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    run_id,
    signal,
    AVG(value) as avg_value,
    MIN(value) as min_value,
    MAX(value) as max_value,
    COUNT(*) as sample_count
FROM telemetry
GROUP BY bucket, run_id, signal
WITH NO DATA;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('telemetry_1min',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE);

-- Insert default organization and admin user
INSERT INTO orgs (id, name) 
VALUES ('00000000-0000-0000-0000-000000000001', 'Demo Organization')
ON CONFLICT DO NOTHING;

INSERT INTO users (id, org_id, email, name, role)
VALUES (
    '00000000-0000-0000-0000-000000000002',
    '00000000-0000-0000-0000-000000000001',
    'admin@demo.local',
    'Demo Admin',
    'Admin'
)
ON CONFLICT (email) DO NOTHING;

-- Insert sample components
INSERT INTO components (org_id, kind, vendor, part_number, params) VALUES
('00000000-0000-0000-0000-000000000001', 'MOSFET', 'Infineon', 'AIMW120R045M1', 
 '{"technology": "SiC", "Vds": 1200, "Rds_on": 0.045, "Id_max": 36, "Qg": 67, "package": "TO247-4"}'),
('00000000-0000-0000-0000-000000000001', 'MOSFET', 'Wolfspeed', 'C3M0075120K',
 '{"technology": "SiC", "Vds": 1200, "Rds_on": 0.075, "Id_max": 30, "Qg": 51, "package": "TO247-4"}'),
('00000000-0000-0000-0000-000000000001', 'MOSFET', 'GaN Systems', 'GS66516T',
 '{"technology": "GaN", "Vds": 650, "Rds_on": 0.025, "Id_max": 60, "Qg": 12, "package": "GaNPX"}'),
('00000000-0000-0000-0000-000000000001', 'INDUCTOR', 'Coilcraft', 'XAL1580-223',
 '{"L": 22e-6, "I_sat": 28, "I_rms": 23, "DCR": 0.0041, "core": "Composite"}'),
('00000000-0000-0000-0000-000000000001', 'CAPACITOR', 'TDK', 'C5750X7S2A106M',
 '{"C": 10e-6, "V_rated": 100, "ESR": 0.002, "I_ripple": 8.5, "dielectric": "X7S"}')
ON CONFLICT (vendor, part_number) DO NOTHING;

-- Create function for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_orgs_updated_at BEFORE UPDATE ON orgs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_designs_updated_at BEFORE UPDATE ON designs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_components_updated_at BEFORE UPDATE ON components
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();