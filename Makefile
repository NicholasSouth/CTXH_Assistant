.PHONY: setup demo eval

setup:
	uv sync
	@test -f .env || cp .env.example .env

demo:
	uv run python ctxh_agent.py

eval:
	uv run eval.py
