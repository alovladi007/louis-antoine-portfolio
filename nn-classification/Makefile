.PHONY: help install test train evaluate clean docker run-demo lint format

help:
	@echo "Neural Networks Classification Suite - Makefile Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install       Install all dependencies"
	@echo "  make install-dev   Install with development dependencies"
	@echo ""
	@echo "Training & Evaluation:"
	@echo "  make train         Train model with default config"
	@echo "  make evaluate      Evaluate trained model"
	@echo "  make benchmark     Run performance benchmarks"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run all tests"
	@echo "  make lint          Run code linting"
	@echo "  make format        Format code with black"
	@echo "  make clean         Clean generated files"
	@echo ""
	@echo "Deployment:"
	@echo "  make docker        Build Docker image"
	@echo "  make export        Export model to ONNX"
	@echo "  make optimize      Optimize model for deployment"
	@echo ""
	@echo "Demo:"
	@echo "  make run-demo      Start interactive web demo"
	@echo "  make notebook      Start Jupyter notebook server"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	python -m ipykernel install --user --name nn-classification

install-dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

train:
	python scripts/train.py \
		--config configs/default.yaml \
		--data-path data/ \
		--output-dir checkpoints/ \
		--epochs 50 \
		--batch-size 32

evaluate:
	python scripts/evaluate.py \
		--model-path checkpoints/best_model.pth \
		--data-path data/test/ \
		--batch-size 64

benchmark:
	python scripts/benchmark.py \
		--models cnn-small cnn-cbam resnet18 vit-tiny \
		--metrics accuracy latency memory \
		--output reports/benchmark.json

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -m "not slow"

lint:
	flake8 src/ tests/ scripts/
	pylint src/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	rm -rf checkpoints/*.pth logs/* figures/*

docker:
	docker build -t nn-classification:latest .
	@echo "Docker image built: nn-classification:latest"

docker-run:
	docker run -it --rm \
		--gpus all \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/checkpoints:/app/checkpoints \
		-p 8888:8888 \
		nn-classification:latest

export:
	python scripts/export_model.py \
		--model-path checkpoints/best_model.pth \
		--export-format onnx \
		--output-path deployment/model.onnx

optimize:
	python scripts/optimize_model.py \
		--model-path checkpoints/best_model.pth \
		--optimization quantization \
		--target-device mobile \
		--output-path deployment/model_optimized.pth

run-demo:
	@echo "Starting Neural Networks Classification Demo..."
	@echo "Opening browser at http://localhost:8000"
	python -m http.server 8000 --directory ../

notebook:
	jupyter notebook notebooks/ --ip=0.0.0.0 --port=8888 --no-browser

download-data:
	@echo "Downloading datasets..."
	python scripts/download_data.py --dataset semiconductor --output data/semiconductor/
	python scripts/download_data.py --dataset medical --output data/medical/

tensorboard:
	tensorboard --logdir logs/ --port 6006

profile:
	python scripts/profile_model.py \
		--model-path checkpoints/best_model.pth \
		--input-size 224 \
		--batch-size 1 \
		--device cuda

validate:
	@echo "Validating project structure..."
	python scripts/validate_project.py
	@echo "✓ All checks passed!"

# Development shortcuts
dev: install-dev lint test

# Full pipeline
all: clean install train evaluate test

# CI/CD pipeline
ci: lint test validate

# Production deployment
deploy: clean optimize export docker
	@echo "Model ready for deployment!"