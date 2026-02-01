.PHONY: help up up-obs down logs ingest api ui eval debug

help:
	@echo "Targets:"
	@echo "  up       - Start core services (ollama, qdrant, api, ui)"
	@echo "  up-obs   - Start core + observability stack (langfuse, dbs)"
	@echo "  down     - Stop all services"
	@echo "  logs     - Tail logs"
	@echo "  ingest   - Run ingestion via API"
	@echo "  api      - Tail API logs"
	@echo "  ui       - Tail UI logs"
	@echo "  eval     - Run RAGAS evaluation inside api container"
	@echo "  eval-router - Run router tool selection evaluation"
	@echo "  lint    - Run ruff lint"
	@echo "  test    - Run pytest smoke tests"
	@echo "  prompt-test - Run prompt regression tests"
	@echo "  debug    - Start api with debugpy and wait for VS Code attach"

up:
	docker compose up -d

up-obs:
	docker compose --profile obs up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

ingest:
	curl -fsS -X POST http://localhost:8000/ingest

api:
	docker compose logs -f --tail=200 api

ui:
	docker compose logs -f --tail=200 ui

eval:
	docker compose exec api env PYTHONPATH=/app python app/eval/ragas_eval.py

eval-router:
	docker compose exec api env PYTHONPATH=/app python app/eval/router_eval.py

lint:
	ruff check .

test:
	pytest -q

prompt-test:
	docker compose exec api npx --yes promptfoo eval

debug:
	DEBUGPY=1 docker compose up api -d
