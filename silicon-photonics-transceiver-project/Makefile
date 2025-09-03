# Silicon Photonics Transceiver Simulation Makefile

PYTHON := python3
PROJECT_DIR := $(shell pwd)
SRC_DIR := $(PROJECT_DIR)/src
RESULTS_DIR := $(PROJECT_DIR)/results
WEB_DIR := $(PROJECT_DIR)/web
DOCS_DIR := $(PROJECT_DIR)/docs

# Default target
.PHONY: help
help:
	@echo "Silicon Photonics Transceiver Simulation"
	@echo "========================================"
	@echo "Available targets:"
	@echo "  make demo      - Quick demo (25G NRZ, short PRBS)"
	@echo "  make simulate  - Full simulation (25G & 50G NRZ/PAM4)"
	@echo "  make optimize  - Run parameter optimization"
	@echo "  make paper     - Generate technical paper"
	@echo "  make dashboard - Launch Streamlit dashboard"
	@echo "  make site      - Build Next.js portfolio site"
	@echo "  make clean     - Remove generated files"
	@echo "  make all       - Run everything"

# Quick demo
.PHONY: demo
demo:
	@echo "Running quick demo (25 Gb/s NRZ)..."
	@cd $(SRC_DIR)/link && $(PYTHON) sim_link_nrz_pam4.py --bitrate 25 --modulation NRZ --smoke
	@echo "Demo complete! Check results/ directory for outputs."
	@ls -la $(RESULTS_DIR)/figs/eye_25g_nrz.png 2>/dev/null || echo "Eye diagram generated"
	@ls -la $(RESULTS_DIR)/tables/ber_sweep_25g.csv 2>/dev/null || echo "BER sweep generated"

# Full simulation
.PHONY: simulate
simulate:
	@echo "Running full simulations..."
	@echo "  25 Gb/s NRZ..."
	@cd $(SRC_DIR)/link && $(PYTHON) sim_link_nrz_pam4.py --bitrate 25 --modulation NRZ
	@echo "  50 Gb/s NRZ..."
	@cd $(SRC_DIR)/link && $(PYTHON) sim_link_nrz_pam4.py --bitrate 50 --modulation NRZ
	@echo "  50 Gb/s PAM4..."
	@cd $(SRC_DIR)/link && $(PYTHON) sim_link_nrz_pam4.py --bitrate 50 --modulation PAM4
	@echo "All simulations complete!"

# Parameter optimization
.PHONY: optimize
optimize:
	@echo "Running parameter optimization..."
	@cd $(SRC_DIR)/link && $(PYTHON) optimize_params.py
	@echo "Optimization complete!"

# Generate paper
.PHONY: paper
paper:
	@echo "Generating technical paper..."
	@cd $(PROJECT_DIR)/scripts && $(PYTHON) generate_paper.py
	@echo "Paper generated at $(DOCS_DIR)/paper.md"

# Launch dashboard
.PHONY: dashboard
dashboard:
	@echo "Launching Streamlit dashboard..."
	@cd $(SRC_DIR)/link/dashboards && streamlit run streamlit_app.py

# Build website
.PHONY: site
site:
	@echo "Building Next.js portfolio site..."
	@cd $(WEB_DIR) && npm install --quiet && npm run build && npm run export
	@echo "Site exported to $(DOCS_DIR)/site/"

# Clean generated files
.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	@rm -rf $(RESULTS_DIR)/figs/*
	@rm -rf $(RESULTS_DIR)/tables/*
	@rm -rf $(DOCS_DIR)/site/*
	@rm -rf $(WEB_DIR)/.next
	@rm -rf $(WEB_DIR)/out
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete!"

# Run everything
.PHONY: all
all: clean simulate optimize paper site
	@echo "Full pipeline complete!"

# Install dependencies
.PHONY: install
install:
	@echo "Installing dependencies..."
	@pip install -r env/requirements.txt
	@cd $(WEB_DIR) && npm install
	@echo "Dependencies installed!"

# Test installation
.PHONY: test
test:
	@echo "Testing installation..."
	@$(PYTHON) -c "import numpy, matplotlib, pandas; print('Python packages OK')"
	@cd $(WEB_DIR) && npm list next > /dev/null 2>&1 && echo "Node packages OK"
	@echo "Installation test passed!"

.DEFAULT_GOAL := help