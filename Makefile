.PHONY: setup run calibrate test lint package clean

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"

run:
	$(PYTHON) apps/edge_service.py --config configs/site-default.yaml

calibrate:
	$(PYTHON) apps/calibrate.py

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	@echo "Linting with ruff (if installed)..."
	@which ruff > /dev/null && ruff check . || echo "ruff not installed, skipping"
	@echo "Formatting with black (if installed)..."
	@which black > /dev/null && black --check . || echo "black not installed, skipping"

package:
	tar -czf cups-counter-$(shell date +%Y%m%d).tar.gz \
		--exclude='$(VENV)' \
		--exclude='data/*' \
		--exclude='*.pyc' \
		--exclude='__pycache__' \
		--exclude='.git' \
		.

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf $(VENV)

