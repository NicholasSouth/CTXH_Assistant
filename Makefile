setup:
  uv sync
  @test -f .env || cp .env.example .env
	@test -f data/index.json || uv run python build_index.py
demo:
  uv run python ctxh_agent.py
eval:
  uv run eval.py
