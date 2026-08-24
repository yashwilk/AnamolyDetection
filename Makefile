.PHONY: install lint format typecheck test pipeline report serve docker-build

install:
	pip install -r requirements-dev.txt

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy src app

test:
	pytest --cov=src --cov=app --cov-report=term-missing

pipeline:
	python scripts/run_pipeline.py --stages generate validate train

report:
	python scripts/generate_report_charts.py

serve:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t anomaly-detection .
