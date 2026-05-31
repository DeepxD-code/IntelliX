.PHONY: install test run docker-build docker-run lint clean

install:
	pip install -r requirements.txt

test:
	python -X utf8 -m pytest test_profiler.py test_app.py -v --tb=short

run:
	python -X utf8 app.py

docker-build:
	docker compose build

docker-run:
	docker compose up -d

lint:
	ruff check profiler_main.py app.py test_profiler.py test_app.py

lint-fix:
	ruff check --fix profiler_main.py app.py test_profiler.py test_app.py

clean:
	rm -rf __pycache__ models/ logs/
	rm -f profile_results.json test_results.json
