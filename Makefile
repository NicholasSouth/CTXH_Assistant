setup:
  uv sync
demo:
  uv run python ctxh_agent.py
eval:
  uv run eval.py --run 3
