# MediMetrics Runbook

## Quick Reference

### Service URLs
- **Web App**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Inference**: http://localhost:9200
- **Orthanc**: http://localhost:8042
- **Grafana**: http://localhost:3001
- **Prometheus**: http://localhost:9090
- **MinIO**: http://localhost:9001

### Default Credentials
- **Demo Admin**: admin@demo.local / Demo123!
- **Orthanc**: orthanc / orthanc
- **Grafana**: admin / admin
- **MinIO**: medimetrics / medimetricssecret

## Operations Guide

### Starting the System

```bash
# Full stack startup
make init          # First time only
make dev          # Development mode
make dev-detached # Background mode

# Individual services
docker compose up postgres redis minio  # Core services
docker compose up api                   # API only
docker compose up web                   # Web only
docker compose up inference worker      # ML services
```

### Stopping the System

```bash
make stop         # Graceful shutdown
docker compose down -v  # Full cleanup (removes volumes)
```

### Health Checks

```bash
# Check all services
make status

# Individual health endpoints
curl http://localhost:8000/health  # API
curl http://localhost:9200/health  # Inference
curl http://localhost:3000/api/health  # Web
```

## Common Tasks

### Database Operations

```bash
# Run migrations
make migrate

# Seed demo data
make seed

# Database shell
make shell-db

# Backup database
docker compose exec postgres pg_dump -U postgres medimetrics > backup.sql

# Restore database
docker compose exec -T postgres psql -U postgres medimetrics < backup.sql
```

### User Management

```bash
# Create admin user via API
curl -X POST http://localhost:8000/users \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "newadmin@example.com",
    "password": "SecurePass123!",
    "role": "ADMIN",
    "firstName": "New",
    "lastName": "Admin"
  }'

# Reset user password
docker compose exec api npm run cli -- reset-password user@example.com
```

### File Storage

```bash
# Access MinIO console
open http://localhost:9001

# Create buckets manually
docker compose exec minio mc mb minio/new-bucket

# List files
docker compose exec minio mc ls minio/medimetrics-raw
```

### DICOM Operations

```bash
# Upload DICOM to Orthanc
curl -X POST http://localhost:8042/instances \
  -u orthanc:orthanc \
  --data-binary @sample.dcm

# Query studies
curl http://localhost:8042/studies \
  -u orthanc:orthanc

# Load fixtures
make load-fixtures
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker compose logs <service-name>

# Common issues:
# - Port already in use: Change port in docker-compose.yml
# - Database connection: Ensure postgres is healthy
# - Missing env vars: Check .env file
```

### Database Issues

```bash
# Reset database
docker compose down postgres
docker volume rm medimetrics-nest_postgres_data
docker compose up postgres

# Fix migrations
docker compose exec api npm run migration:revert
docker compose exec api npm run migration:run
```

### Authentication Problems

```bash
# Clear sessions
docker compose exec redis redis-cli FLUSHALL

# Generate new JWT secret
openssl rand -base64 32
# Update JWT_SECRET in .env and restart
```

### Storage Issues

```bash
# Check MinIO status
docker compose logs minio

# Recreate buckets
docker compose run --rm minio-client

# Fix permissions
docker compose exec minio chown -R 1000:1000 /data
```

### Inference Service Issues

```bash
# Check worker status
docker compose logs worker

# Restart workers
docker compose restart worker

# Check Redis queue
docker compose exec redis redis-cli
> LLEN inference:queue
> LRANGE inference:queue 0 -1
```

## Monitoring

### Metrics

```bash
# Prometheus metrics
curl http://localhost:9100/metrics  # API
curl http://localhost:9201/metrics  # Inference

# Grafana dashboards
open http://localhost:3001
# Login: admin/admin
# Dashboards → Browse → API/Inference Overview
```

### Logs

```bash
# All logs
make logs

# Service-specific
make logs-api
make logs-web
make logs-inference

# Follow logs
docker compose logs -f <service>

# Export logs
docker compose logs > system.log
```

### Performance

```bash
# Check resource usage
docker stats

# Database queries
docker compose exec postgres psql -U postgres -d medimetrics -c "
  SELECT query, calls, mean_exec_time 
  FROM pg_stat_statements 
  ORDER BY mean_exec_time DESC 
  LIMIT 10;"

# Redis memory
docker compose exec redis redis-cli INFO memory
```

## Maintenance

### Updates

```bash
# Update dependencies
docker compose exec api npm update
docker compose exec web npm update
docker compose exec inference poetry update

# Rebuild images
docker compose build --no-cache

# Apply migrations
make migrate
```

### Backups

```bash
# Automated backup
make backup

# Manual backup
DATE=$(date +%Y%m%d_%H%M%S)
docker compose exec postgres pg_dump -U postgres medimetrics > backups/db_$DATE.sql
docker compose exec minio mc mirror minio/medimetrics-raw backups/s3_$DATE/

# Restore
make restore  # Latest backup
docker compose exec -T postgres psql -U postgres medimetrics < backups/db_20240101_120000.sql
```

### Cleanup

```bash
# Remove old logs
find logs/ -name "*.log" -mtime +30 -delete

# Clean Docker
docker system prune -a --volumes

# Reset everything
make reset
```

## Security Operations

### Certificate Management

```bash
# Generate self-signed cert (development)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Check certificate expiry
openssl x509 -in cert.pem -noout -dates
```

### Secret Rotation

```bash
# Generate new secrets
openssl rand -base64 32  # JWT_SECRET
openssl rand -base64 32  # FIELD_ENCRYPTION_KEK
openssl rand -base64 32  # WEBHOOK_HMAC_SECRET

# Update .env and restart services
docker compose restart api inference
```

### Audit Logs

```bash
# View recent audit entries
curl http://localhost:8000/admin/audit?limit=100 \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Export audit logs
docker compose exec api npm run export-audit -- --start 2024-01-01 --end 2024-12-31
```

## Incident Response

### High CPU Usage

1. Identify the service: `docker stats`
2. Check logs: `docker compose logs <service>`
3. Restart if needed: `docker compose restart <service>`
4. Scale workers: `docker compose up --scale worker=4`

### Memory Leak

1. Monitor memory: `docker stats`
2. Check for large queries: Database logs
3. Clear caches: `docker compose exec redis redis-cli FLUSHALL`
4. Restart service: `docker compose restart <service>`

### Service Outage

1. Check health: `make status`
2. Review logs: `make logs`
3. Restart services: `docker compose restart`
4. Verify data integrity: Run health checks
5. Notify users if needed

## Production Deployment

### Pre-deployment Checklist

- [ ] Change all default passwords
- [ ] Generate new secret keys
- [ ] Configure TLS certificates
- [ ] Set up backup strategy
- [ ] Configure monitoring alerts
- [ ] Review security settings
- [ ] Test disaster recovery
- [ ] Document custom configurations

### Environment Variables

```bash
# Production .env template
NODE_ENV=production
WEB_BASE_URL=https://medimetrics.example.com
API_BASE_URL=https://api.medimetrics.example.com

# Use external services
POSTGRES_HOST=rds.amazonaws.com
REDIS_URL=redis://elasticache.amazonaws.com:6379
S3_ENDPOINT=https://s3.amazonaws.com

# Strong secrets (generate with openssl rand -base64 32)
JWT_SECRET=<generated>
FIELD_ENCRYPTION_KEK=<generated>
WEBHOOK_HMAC_SECRET=<generated>

# Security
RATE_LIMIT=50/minute
CORS_ORIGINS=https://medimetrics.example.com
ENABLE_2FA=true
```

### Deployment Commands

```bash
# Docker Compose production
docker compose --profile prod up -d

# Kubernetes
kubectl apply -k k8s/overlays/prod

# Health verification
./scripts/health_check.sh production
```

## Support

### Getting Help

1. Check logs first: `make logs`
2. Review this runbook
3. Search known issues: GitHub Issues
4. Contact support: support@medimetrics.example

### Reporting Issues

Include:
- Service logs
- Error messages
- Steps to reproduce
- Environment details
- Screenshots if applicable

---

**Last Updated**: 2024
**Version**: 1.0.0