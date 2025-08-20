.PHONY: help init dev prod test clean reset logs

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
RED := \033[0;31m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "${CYAN}MediMetrics - Medical Image Analysis Platform${NC}"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "${GREEN}%-15s${NC} %s\n", $$1, $$2}'

init: ## Initialize the project (install deps, setup DB, seed data)
	@echo "${CYAN}Initializing MediMetrics...${NC}"
	@echo "${YELLOW}Setting up environment...${NC}"
	@test -f .env || cp .env.example .env
	@echo "${YELLOW}Creating Docker networks and volumes...${NC}"
	@docker network create medimetrics 2>/dev/null || true
	@echo "${YELLOW}Building containers...${NC}"
	@docker compose build
	@echo "${YELLOW}Starting core services...${NC}"
	@docker compose up -d postgres redis minio orthanc
	@echo "${YELLOW}Waiting for services...${NC}"
	@sleep 10
	@echo "${YELLOW}Creating S3 buckets...${NC}"
	@docker compose run --rm minio-client
	@echo "${YELLOW}Running database migrations...${NC}"
	@docker compose run --rm api npm run migration:run
	@echo "${YELLOW}Seeding demo data...${NC}"
	@docker compose run --rm api npm run seed
	@echo "${GREEN}✓ Initialization complete!${NC}"

dev: ## Start development environment
	@echo "${CYAN}Starting development environment...${NC}"
	@docker compose up --build

dev-detached: ## Start development environment in background
	@echo "${CYAN}Starting development environment (detached)...${NC}"
	@docker compose up -d --build
	@echo "${GREEN}✓ Services started!${NC}"
	@echo ""
	@echo "Access points:"
	@echo "  Web App:    ${CYAN}http://localhost:3000${NC} (admin@demo.local / Demo123!)"
	@echo "  API Docs:   ${CYAN}http://localhost:8000/docs${NC}"
	@echo "  Orthanc:    ${CYAN}http://localhost:8042${NC} (orthanc/orthanc)"
	@echo "  Grafana:    ${CYAN}http://localhost:3001${NC} (admin/admin)"
	@echo "  MinIO:      ${CYAN}http://localhost:9001${NC} (medimetrics/medimetricssecret)"

prod: ## Start production environment
	@echo "${CYAN}Starting production environment...${NC}"
	@docker compose --profile prod up -d
	@echo "${GREEN}✓ Production services started!${NC}"

stop: ## Stop all services
	@echo "${YELLOW}Stopping services...${NC}"
	@docker compose down
	@echo "${GREEN}✓ Services stopped${NC}"

restart: ## Restart all services
	@echo "${YELLOW}Restarting services...${NC}"
	@docker compose restart
	@echo "${GREEN}✓ Services restarted${NC}"

logs: ## Show logs for all services
	@docker compose logs -f

logs-api: ## Show API logs
	@docker compose logs -f api

logs-web: ## Show web app logs
	@docker compose logs -f web

logs-inference: ## Show inference service logs
	@docker compose logs -f inference worker

test: ## Run all tests
	@echo "${CYAN}Running tests...${NC}"
	@echo "${YELLOW}API tests...${NC}"
	@docker compose exec api npm test
	@echo "${YELLOW}Inference tests...${NC}"
	@docker compose exec inference pytest
	@echo "${YELLOW}E2E tests...${NC}"
	@docker compose exec web npm run test:e2e
	@echo "${GREEN}✓ All tests passed!${NC}"

test-api: ## Run API tests only
	@docker compose exec api npm test

test-inference: ## Run inference tests only
	@docker compose exec inference pytest

test-e2e: ## Run E2E tests only
	@docker compose exec web npm run test:e2e

lint: ## Run linters
	@echo "${CYAN}Running linters...${NC}"
	@docker compose exec api npm run lint
	@docker compose exec web npm run lint
	@docker compose exec inference flake8
	@echo "${GREEN}✓ Linting complete${NC}"

format: ## Format code
	@echo "${CYAN}Formatting code...${NC}"
	@docker compose exec api npm run format
	@docker compose exec web npm run format
	@docker compose exec inference black src/
	@echo "${GREEN}✓ Formatting complete${NC}"

migrate: ## Run database migrations
	@echo "${YELLOW}Running migrations...${NC}"
	@docker compose exec api npm run migration:run
	@echo "${GREEN}✓ Migrations complete${NC}"

seed: ## Seed demo data
	@echo "${YELLOW}Seeding demo data...${NC}"
	@docker compose exec api npm run seed
	@echo "${GREEN}✓ Seeding complete${NC}"

load-fixtures: ## Load sample DICOM files
	@echo "${YELLOW}Loading DICOM fixtures...${NC}"
	@bash scripts/load_fixtures.sh
	@echo "${GREEN}✓ Fixtures loaded${NC}"

shell-api: ## Open API shell
	@docker compose exec api sh

shell-web: ## Open web shell
	@docker compose exec web sh

shell-db: ## Open database shell
	@docker compose exec postgres psql -U postgres -d medimetrics

clean: ## Clean build artifacts and caches
	@echo "${YELLOW}Cleaning build artifacts...${NC}"
	@docker compose exec api rm -rf dist node_modules
	@docker compose exec web rm -rf .next node_modules
	@docker compose exec inference rm -rf __pycache__ .pytest_cache
	@echo "${GREEN}✓ Clean complete${NC}"

reset: ## Reset everything (WARNING: deletes all data!)
	@echo "${RED}WARNING: This will delete all data!${NC}"
	@echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
	@sleep 5
	@echo "${YELLOW}Resetting...${NC}"
	@docker compose down -v
	@docker network rm medimetrics 2>/dev/null || true
	@rm -f .env
	@echo "${GREEN}✓ Reset complete${NC}"

status: ## Show service status
	@echo "${CYAN}Service Status:${NC}"
	@docker compose ps

metrics: ## Open metrics endpoints
	@echo "${CYAN}Metrics Endpoints:${NC}"
	@echo "  API:        ${CYAN}http://localhost:9100/metrics${NC}"
	@echo "  Inference:  ${CYAN}http://localhost:9201/metrics${NC}"
	@echo "  Prometheus: ${CYAN}http://localhost:9090${NC}"

backup: ## Backup database and files
	@echo "${YELLOW}Creating backup...${NC}"
	@mkdir -p backups
	@docker compose exec postgres pg_dump -U postgres medimetrics > backups/db_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "${GREEN}✓ Backup created in backups/${NC}"

restore: ## Restore from latest backup
	@echo "${YELLOW}Restoring from latest backup...${NC}"
	@docker compose exec -T postgres psql -U postgres medimetrics < $(shell ls -t backups/*.sql | head -1)
	@echo "${GREEN}✓ Restore complete${NC}"

version: ## Show version information
	@echo "${CYAN}MediMetrics Version Information:${NC}"
	@echo "API Version:       $(shell docker compose exec api npm version --json | grep medimetrics-api)"
	@echo "Web Version:       $(shell docker compose exec web npm version --json | grep medimetrics-web)"
	@echo "Docker Compose:    $(shell docker compose version)"
	@echo "Docker:            $(shell docker version --format '{{.Server.Version}}')"

# Default target
.DEFAULT_GOAL := help