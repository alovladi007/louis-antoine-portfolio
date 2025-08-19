# PCM GST Research Makefile

.PHONY: help install test run clean docker bundle all

help:
	@echo "PCM GST Research - Available targets:"
	@echo "  install  - Install Python dependencies"
	@echo "  test     - Run unit tests"
	@echo "  run      - Run complete simulation pipeline"
	@echo "  clean    - Clean generated files"
	@echo "  bundle   - Create ZIP bundle of project"
	@echo "  docker   - Build Docker image"
	@echo "  all      - Install, test, and run everything"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

run:
	python run_simulation.py

clean:
	rm -rf data/*.csv data/*.png
	rm -rf reports/figures/*.png
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

bundle:
	python -c "from pcm.utils import create_project_bundle; create_project_bundle()"

docker:
	docker build -t pcm-gst-research .

all: install test run
	@echo "Complete pipeline executed successfully!"