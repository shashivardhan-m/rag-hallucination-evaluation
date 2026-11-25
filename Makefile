.PHONY: test validate benchmark reproduce clean

PYTHON ?= python3

test:
	cd backend && $(PYTHON) -m pytest tests/ -v

validate:
	$(PYTHON) backend/scripts/validate_judge.py

benchmark:
	$(PYTHON) backend/scripts/run_benchmark.py

reproduce: validate benchmark
	@echo "=========================================================="
	@echo "All benchmark and judge validation results reproduced!"
	@echo "Reports saved to backend/data/benchmarks/ & backend/data/experiments/"
	@echo "=========================================================="

clean:
	rm -rf backend/data/cache/*.jsonl
	rm -rf backend/.pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
