import asyncio
import json
import os
import random
import sys
import time
import re
import argparse
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import (
    Groq, APIConnectionError, APIStatusError,
    RateLimitError, AuthenticationError, BadRequestError
)
from mcp import StdioServerParameters, Client

load_dotenv()

_HERE = Path(__file__).resolve().parent
SERVER_PATH = _HERE / "ctxh_server.py"
CALL_TIMEOUT = 30

LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
LLM_TIMEOUT = 60
API_LOG = _HERE / "logs" / "api_log.jsonl"
PROMPT_PATH = _HERE / "template" / "system_prompt.txt"

MAX_STEPS = 8            
OBS_MAX_CHARS = 6000     # Prevent overly long Observations from bloating the prompt
MAX_HISTORY_TURNS = 5    # Keep only the last 5 turns (question + final answer)
CONTEXT_TOOLS = {"search_documents", "get_document", "find_activity"}
WRITE_TOOLS = {"register_activity"} 
LOG_DIR = _HERE / "logs"
CONTEXT_CHUNK_CHARS = 800

_groq: Groq | None = None

@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency: float

def _log_api(entry: dict) -> None:
    API_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now().isoformat(), **entry}
    with open(API_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def call_llm(messages: list[dict], model: str = LLM_MODEL,
             max_retries: int = 3) -> LLMResult:
    global _groq
    if _groq is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Missing GROQ_API_KEY: copy .env.example to .env and fill in your key.")
        _groq = Groq(api_key=key, timeout=LLM_TIMEOUT)

    for attempt in range(1, max_retries + 2):
        start = time.time()
        try:
            resp = _groq.chat.completions.create(model=model, messages=messages)
        except (AuthenticationError, BadRequestError) as e:
            _log_api({"model": model, "success": False, "attempt": attempt,
                      "error": type(e).__name__, "status": e.status_code})
            if isinstance(e, AuthenticationError):
                print("Authentication failed. Check GROQ_API_KEY.", file=sys.stderr)
            else:
                print(f"Bad request: {e}", file=sys.stderr)
            raise
        except (RateLimitError, APIConnectionError, APIStatusError) as e:
            status = getattr(e, "status_code", None)
            retryable = (isinstance(e, (RateLimitError, APIConnectionError))
                         or (status is not None and 500 <= status < 600))
            _log_api({"model": model, "success": False, "attempt": attempt,
                      "error": type(e).__name__, "status": status})
            if not retryable or attempt > max_retries:
                raise
            delay = 2 ** (attempt - 1) + random.uniform(0, 1)
            print(f"{type(e).__name__}. Retrying in {delay:.1f}s...", file=sys.stderr)
            time.sleep(delay)
            continue

        elapsed = round(time.time() - start, 2)
        result = LLMResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            latency=elapsed,
        )
        _log_api({"model": model, "success": True, "attempt": attempt,
                  "prompt_tokens": result.prompt_tokens,
                  "completion_tokens": result.completion_tokens,
                  "latency_seconds": elapsed})
        return result

@dataclass
class ToolParam:
    name: str
    type: str
    required: bool
    default: object = None
    has_default: bool = False

@dataclass
class ToolSpec:
    name: str
    description: str
    params: list[ToolParam]

def parse_params(schema: dict) -> list[ToolParam]:
    """Convert the server's JSON schema into dataclasses"""
    required = set(schema.get("required", []))
    return [ToolParam(name, p.get("type", "any"), name in required,
                      p.get("default"), "default" in p)
            for name, p in schema.get("properties", {}).items()]

def render_tools(specs: list[ToolSpec]) -> str:
    blocks = []
    for t in specs:
        params = []
        for p in t.params:
            s = f"{p.name}: {p.type}"
            if not p.required and p.has_default:
                s += f" = {json.dumps(p.default, ensure_ascii=False)}"
            params.append(s)
        blocks.append(f"- {t.name}({', '.join(params)})\n  {' '.join(t.description.split())}")
    return "\n".join(blocks)

def build_system_prompt(specs: list[ToolSpec]) -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").replace("{tools}", render_tools(specs))

@dataclass
class Parsed:
    kind: str # "final" or "action" or "error"
    text: str = ""                
    name: str = ""
    args: dict | None = None

def parse_reply(reply: str) -> Parsed:
    reply = re.split(r"\n\s*Observation:", reply, maxsplit=1)[0]

    m_act = re.search(r"^\s*Action:\s*(\w+)\s*(.*)$", reply, flags=re.MULTILINE | re.DOTALL)
    m_fin = re.search(r"Final Answer:\s*(.+)", reply, flags=re.IGNORECASE | re.DOTALL)

    if m_act and (not m_fin or m_act.start() < m_fin.start()):
        name, rest = m_act.group(1), m_act.group(2).strip()
        if not rest:
            return Parsed("action", name=name, args={})
        try:
            args, _ = json.JSONDecoder().raw_decode(rest)   
        except json.JSONDecodeError as e:
            return Parsed("error", text=f"Error: arguments of '{name}' are not valid JSON ({e.msg}). "
                                        f"Use the format: Action: {name} {{\"arg\": \"value\"}}")
        if not isinstance(args, dict):
            return Parsed("error", text="Error: arguments must be a JSON object, e.g. {\"query\": \"...\"}")
        return Parsed("action", name=name, args=args)

    if m_fin:
        return Parsed("final", text=m_fin.group(1).strip())
    return Parsed("error", text="Error: reply must contain either 'Action: tool_name {json}' "
                                "or 'Final Answer: ...'.")

def validate_action(specs: list[ToolSpec], name: str, args: dict) -> str | None:
    spec = next((t for t in specs if t.name == name), None)
    if spec is None:
        return f"Error: unknown tool '{name}'. Available: {', '.join(t.name for t in specs)}"
    names = [p.name for p in spec.params]
    missing = [p.name for p in spec.params if p.required and p.name not in args]
    if missing:
        return f"Error: missing argument(s) {missing} for '{name}'."
    extra = [k for k in args if k not in names]
    if extra:
        return f"Error: unknown argument(s) {extra} for '{name}'. Allowed: {names}."
    return None

@dataclass
class Step:
    index: int
    thought: str
    tool: str | None          # None if the step is Final Answer
    args: dict | None
    observation: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0      # LLM call time for only this step
    confirmed: bool | None = None
 
@dataclass
class RunResult:
    question: str
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)   
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0      # Total time for the whole turn (seconds)
    finished: bool = False    # True if the model produces the Final Answer on its own
    error: str | None = None  # System error (e.g. LLM is down), None if everything is normal
    model: str = LLM_MODEL

def _extract_thought(reply: str) -> str:
    m = re.search(r"Thought:\s*(.*?)(?=\n\s*(?:Action|Final Answer):|\Z)",
                  reply, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""
 
def _is_useful_context(obs: str) -> bool:
    return not obs.startswith("Error") and not obs.startswith("No ")

def split_contexts(tool: str, obs: str) -> list[str]:
    """Split the Observation into smaller contexts, each context is a separate chunk."""
    if tool == "search_documents":
        # Search results: chunks separated by blank lines
        parts = [p.strip() for p in re.split(r"\n\n(?=DOC_\d+ \[)", obs)]
    elif tool == "find_activity":
        # Each line is an activity code
        parts = [p.strip() for p in obs.split("\n")]
    else:
        # get_document: long text, group lines into blocks of ~800 characters
        parts, buf = [], ""
        for line in obs.split("\n"):
            if buf and len(buf) + len(line) > CONTEXT_CHUNK_CHARS:
                parts.append(buf.strip())
                buf = ""
            buf += line + "\n"
        parts.append(buf.strip())
    return [p for p in parts if p]

def extend_history(history: list[dict], question: str, result: RunResult) -> list[dict]:
    """Condensed history: only the question + Final Answer. Failed turns are excluded."""
    if not result.finished:
        return history
    
    new_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": result.answer},
    ]
    return new_history[-2 * MAX_HISTORY_TURNS:]

class CtxhClient:
    def __init__(self, server_path: Path = SERVER_PATH):
        self._params = StdioServerParameters(
            command="uv",
            args=["run", str(server_path)]
        )
        self._client: Client | None = None

    async def __aenter__(self):
        self._client = Client(self._params)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc):
        await self._client.__aexit__(*exc)

    async def list_tools(self) -> list[ToolSpec]:
        tools = (await self._client.list_tools()).tools
        return [ToolSpec(t.name, t.description or "", parse_params(t.input_schema)) for t in tools]

    async def call_tool(self, name: str, args: dict) -> str:
        try:
            result = await asyncio.wait_for(self._client.call_tool(name, args),
                                            timeout = CALL_TIMEOUT)
        except asyncio.TimeoutError:
            return f"Error: tool '{name}' timed out after {CALL_TIMEOUT}s"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

        text = "\n".join(
            block.text for block in (result.content or []) if getattr(block, "text", None)
        )
        if result.is_error:
            return text if text.startswith("Error") else f"Error: {text}"
        return text

async def run_agent(client: CtxhClient, specs: list[ToolSpec], question: str,
                    history: list[dict] | None = None, max_steps: int = MAX_STEPS, verbose: bool = False
                    , confirm=None) -> RunResult:
    t0 = time.time()
    result = RunResult(question=question)
    messages: list[dict] = [{"role": "system", "content": build_system_prompt(specs)}]
    messages += history or []
    messages.append({"role": "user", "content": question})
 
    for i in range(1, max_steps + 1):
        # call_llm runs in a thread so it doesn't block the MCP client's event loop
        # because call_llm is a synchronous function
        try:
            llm = await asyncio.to_thread(call_llm, messages)
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            result.answer = "Sorry, the system encountered an error while calling the model. Please try again later."
            break
 
        result.prompt_tokens += llm.prompt_tokens
        result.completion_tokens += llm.completion_tokens
 
        # Strip any Observation the model made up on its own
        reply = re.split(r"\n\s*Observation:", llm.text, maxsplit=1)[0].strip()
        parsed = parse_reply(reply)
        step = Step(index=i, thought=_extract_thought(reply), tool=None, args=None,
                    observation="", prompt_tokens=llm.prompt_tokens,
                    completion_tokens=llm.completion_tokens, latency=llm.latency)
 
        if parsed.kind == "final":
            result.steps.append(step)
            result.answer = parsed.text
            result.finished = True
            if verbose:
                print(f"[step {i}]\n"
                      f"Thought: {step.thought}\n"
                      f"Final Answer: {parsed.text[:200]}", file=sys.stderr)
            break
 
        messages.append({"role": "assistant", "content": reply})

        # If the parsed reply has the wrong format, tell the model to fix it
        if parsed.kind == "error":                      
            obs = parsed.text
        else:
            step.tool, step.args = parsed.name, parsed.args

            err = validate_action(specs, parsed.name, parsed.args)
            # Case where the tool not found on the server or has missing / unexpected arguments
            if err:                                     
                obs = err
            else:
                if parsed.name in WRITE_TOOLS and confirm is not None:
                    accept = await asyncio.to_thread(confirm, parsed.name, parsed.args)
                    step.confirmed = accept
                    if accept:
                        obs = await client.call_tool(parsed.name, parsed.args)
                    else:
                        obs = ("Error: user declined this action. Do not retry; "
                               "tell the user it was cancelled.")
                else:
                    obs = await client.call_tool(parsed.name, parsed.args)

                if parsed.name in CONTEXT_TOOLS and _is_useful_context(obs):
                    for c in split_contexts(parsed.name, obs):
                        if c not in result.contexts:      # Remove duplicate chunks across searches
                            result.contexts.append(c)
 
        obs = obs[:OBS_MAX_CHARS]
        step.observation = obs
        result.steps.append(step)
        messages.append({"role": "user", "content": f"Observation: {obs}"})
 
        if verbose:
            action = (f"{step.tool} {json.dumps(step.args, ensure_ascii=False)}"
                      if step.tool else "(parse-error)")
            print(f"[step {i}]\n"
                  f"Thought: {step.thought}\n"
                  f"Action: {action}\n"
                  f"Observation: {obs[:200]!r}", file=sys.stderr)
    else:
        # The loop ran to completion without a break and the model never gave a Final Answer
        result.answer = (f"Xin lỗi, mình chưa tìm được câu trả lời trong giới hạn {max_steps} bước. "
                         "Bạn thử hỏi cụ thể hơn nhé.")
 
    result.latency = round(time.time() - t0, 2)
    return result

def usage_of(result: RunResult) -> dict:
    return {
        "calls": len(result.steps),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
        "max_prompt": max((s.prompt_tokens for s in result.steps), default=0),
        "latency_seconds": result.latency,
        "steps": len(result.steps),
    }

class RunLogger:
    """Writes one logs/run_*.jsonl file per run, one line per question."""

    def __init__(self, log_dir: Path = LOG_DIR):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        self._n = 0

    def log(self, result: RunResult) -> None:
        self._n += 1
        entry = {
            "id": self._n,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": result.question,
            "answer": result.answer,
            "finished": result.finished,
            "error": result.error,
            "model": result.model,
            "usage": usage_of(result),
            "steps": [asdict(s) for s in result.steps],   
            "contexts": result.contexts,                  
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def _print_usage(r: RunResult) -> None:
    print(f"   [steps={len(r.steps)} | tokens={r.prompt_tokens} in + {r.completion_tokens} out "
          f"| {r.latency}s]")

def ask_confirm(name: str, args: dict) -> bool:
    ans = input(f"\n[Xác nhận] Thực hiện {name} {json.dumps(args, ensure_ascii=False)}? (y/n): ")
    return ans.strip().lower() in {"y", "yes"}

async def repl(verbose: bool = False) -> None:
    logger = RunLogger()
    async with CtxhClient() as client:            
        specs = await client.list_tools()
        history: list[dict] = []
        print(f"Chào bạn. Gõ 'exit' hoặc 'quit' để thoát.")
        print(f"Log: {logger.path}")

        while True:
            try:
                question = (await asyncio.to_thread(input, "\nBạn: ")).strip()
            except EOFError:                       
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break

            try:                                   
                result = await run_agent(client, specs, question, history, verbose=verbose, 
                                         confirm=ask_confirm)
            except Exception as e:
                print(f"Unexpected error: {type(e).__name__}: {e}. Please try again later.")
                continue

            history = extend_history(history, question, result)
            print(f"\nTrợ lý: {result.answer}")
            _print_usage(result)

            try:                                   
                logger.log(result)
            except OSError as e:
                print(f"Warning: unable to write log ({e})", file=sys.stderr)

def main() -> None:
    parser = argparse.ArgumentParser(description="Trợ lý CTXH")
    parser.add_argument("--verbose", action="store_true",
                        help="in Thought/Action/Observation của từng bước ra stderr")
    args = parser.parse_args()
    try:
        asyncio.run(repl(verbose=args.verbose))
    except KeyboardInterrupt:
        pass
    except Exception as e:                         
        print("Error: failed to start the assistant. Please try again.", file=sys.stderr)
        if args.verbose:
            print(f"Details: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
            
if __name__ == "__main__":
    main()