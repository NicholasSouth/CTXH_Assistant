import json
import re
import unicodedata
import argparse
import asyncio
from collections import Counter
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
# for tabular writing, building cache key, detect invalid mathematic results like 5/0,
# read JUDGE_MODEL env var, add rando for retry delay, compute standard deviation
import csv, hashlib, math, os, random, statistics

from ctxh_agent import CtxhClient, RunResult, run_agent, LOG_DIR, usage_of

_HERE = Path(__file__).resolve().parent
TESTSET_PATH = _HERE / "tests" / "testset.jsonl"

MIN_QUESTIONS = 12
MIN_NO_ANSWER = 4
VALID_TYPES = {"answerable", "no_answer"}
VALID_SCOPES = {"in_domain", "out_of_domain"}

# Read and validate the testset

def _is_str_list(v) -> bool:
    return isinstance(v, list) and len(v) > 0 and all(isinstance(x, str) and x.strip() for x in v)

def _validate_item(item: dict) -> list[str]:
    """Return the list of errors for one testset line (empty if valid)."""
    errs: list[str] = []

    if not isinstance(item.get("id"), str) or not item["id"].strip():
        errs.append("'id' phải là chuỗi không rỗng")
    if not isinstance(item.get("question"), str) or not item["question"].strip():
        errs.append("'question' phải là chuỗi không rỗng")
    if not _is_str_list(item.get("tools_expected")):
        errs.append("'tools_expected' phải là list chuỗi không rỗng")
    if not isinstance(item.get("ragas"), bool):
        errs.append("'ragas' phải là true/false")

    qtype = item.get("type")
    if qtype not in VALID_TYPES:
        errs.append(f"'type' phải thuộc {sorted(VALID_TYPES)}, nhận được {qtype!r}")
        return errs

    if qtype == "answerable":
        if not isinstance(item.get("reference"), str) or not item["reference"].strip():
            errs.append("câu answerable cần 'reference' là chuỗi không rỗng")
    else:  # no_answer
        if item.get("reference") is not None:
            errs.append("câu no_answer phải có 'reference' = null")
        if item.get("ragas") is not False:
            errs.append("câu no_answer phải có 'ragas' = false")
        if item.get("scope") not in VALID_SCOPES:
            errs.append(f"'scope' phải thuộc {sorted(VALID_SCOPES)}, nhận được {item.get('scope')!r}")
        if "search_documents" not in (item.get("tools_expected") or []):
            errs.append("câu no_answer: 'tools_expected' phải chứa 'search_documents'")
        markers = (item.get("expected") or {}).get("refusal_markers")
        if not _is_str_list(markers):
            errs.append("câu no_answer cần 'expected.refusal_markers' là list chuỗi không rỗng")
    return errs

def _validate_totals(items: list[dict]) -> list[str]:
    errs: list[str] = []
    n_no = sum(1 for it in items if it.get("type") == "no_answer")
    if len(items) < MIN_QUESTIONS:
        errs.append(f"at least {MIN_QUESTIONS} questions is required. Current test set has {len(items)}")
    if n_no < MIN_NO_ANSWER:
        errs.append(f"at least {MIN_NO_ANSWER} questions that is no_answer is required. Current test set has {n_no} questions with no_answer")
    scopes = {it.get("scope") for it in items if it.get("type") == "no_answer"}
    for s in sorted(VALID_SCOPES - scopes):
        errs.append(f"no no_answer question yet with scope '{s}' (at least 1 required per scope)")
    return errs

def load_testset(path: Path = TESTSET_PATH) -> list[dict]:
    """Read the JSONL, validate everything, then report all errors at once. Raise ValueError on errors."""
    errors: list[str] = []
    items: list[dict] = []
    seen: set[str] = set()

    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as e:
                errors.append(f"dòng {lineno}: JSON không hợp lệ ({e.msg})")
                continue
            if not isinstance(item, dict):
                errors.append(f"dòng {lineno}: mỗi dòng phải là một JSON object")
                continue

            errs = _validate_item(item)
            qid = item.get("id")
            if isinstance(qid, str):
                if qid in seen:
                    errs.append(f"trùng id '{qid}'")
                seen.add(qid)
            errors += [f"dòng {lineno} ({qid}): {e}" for e in errs]
            items.append(item)

    errors += _validate_totals(items)
    if errors:
        raise ValueError("Testset không hợp lệ:\n- " + "\n- ".join(errors))
    return items

def summarize_testset(items: list[dict]) -> None:
    by_type = Counter(it["type"] for it in items)
    by_scope = Counter(it["scope"] for it in items if it["type"] == "no_answer")
    n_ragas = sum(1 for it in items if it["ragas"])
    print(f"Testset: {len(items)} câu | answerable={by_type['answerable']} "
          f"no_answer={by_type['no_answer']} | chấm RAGAS={n_ragas}")
    for scope, n in sorted(by_scope.items()):
        print(f"  no_answer/{scope}: {n}")

# Refusal scoring (only for no_answer questions)

def fold(text: str) -> str:
    """Lowercase, strip Vietnamese diacritics, collapse whitespace."""
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text.lower()).strip()

def is_refusal(answer: str, markers: list[str]) -> bool:
    """True if the answer contains at least one marker (matched after normalization)."""
    folded = fold(answer or "")
    return any(fold(m) in folded for m in markers if m.strip())

def _called_tool(result: dict, tool: str) -> bool:
    return any(s.get("tool") == tool for s in result.get("steps", []))

def score_refusal(items: list[dict], results: dict[str, dict]) -> dict:
    """
    items:   list of testset lines (already passed through load_testset).
    results: {question id: one log line} — each log has 'answer', 'finished', 'steps' (each step has 'tool').
    A question passes when: finished=True, search_documents was called, and the answer contains a refusal_marker.
    """
    details: list[dict] = []
    for it in items:
        if it["type"] != "no_answer":
            continue
        res = results.get(it["id"])
        reasons: list[str] = []

        if res is None:
            reasons.append("missing_result")
            answer = ""
        else:
            answer = res.get("answer", "")
            if not res.get("finished"):
                reasons.append("not_finished")
            if not _called_tool(res, "search_documents"):
                reasons.append("no_search")
            if not is_refusal(answer, it["expected"]["refusal_markers"]):
                reasons.append("no_refusal_marker")

        details.append({
            "id": it["id"],
            "scope": it["scope"],
            "passed": not reasons,
            "reasons": reasons,
            "answer": answer,   # for manual reading
        })

    def _rate(rows: list[dict]) -> dict:
        n = len(rows)
        k = sum(1 for r in rows if r["passed"])
        return {"total": n, "passed": k, "rate": round(k / n, 4) if n else None}

    return {
        **_rate(details),
        "by_scope": {s: _rate([r for r in details if r["scope"] == s])
                     for s in sorted({r["scope"] for r in details})},
        "details": details,
    }

# Run the agent on the testset

def _no_write_confirm(name: str, args: dict) -> bool:
    """Eval has no human to press y/n and the testset has no registration question → always decline write actions."""
    return False

def build_entry(item: dict, result: RunResult, run_index: int) -> dict:
    """One eval log line: same structure as RunLogger, plus the testset 'id' and 'run'."""
    return {
        "id": item["id"],
        "run": run_index,
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

def _error_entry(item: dict, run_index: int, exc: Exception) -> dict:
    """Entry for an unexpected error outside run_agent: still has all fields so the scoring steps don't break."""
    return {
        "id": item["id"], "run": run_index,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": item["question"], "answer": "", "finished": False,
        "error": f"{type(exc).__name__}: {exc}", "model": None,
        "usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, 
                  "max_prompt": 0, "latency_seconds": 0.0, "steps": 0},
        "steps": [], "contexts": [],
    }

def _append_jsonl(path: Path, entry: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

async def run_testset(client: CtxhClient, specs: list, items: list[dict],
                      run_index: int, log_path: Path, verbose: bool = False) -> dict[str, dict]:
    """Run the questions one by one (empty history), writing the log right after each question.
    Return {question id: log line} — exactly the shape score_refusal() accepts."""
    results: dict[str, dict] = {}
    for n, item in enumerate(items, start=1):
        try:
            result = await run_agent(client, specs, item["question"], history=[],
                                     verbose=verbose, confirm=_no_write_confirm)
            entry = build_entry(item, result, run_index)
        except Exception as e:            # unexpected error: log it and move on
            entry = _error_entry(item, run_index, e)

        _append_jsonl(log_path, entry)
        results[item["id"]] = entry

        u = entry["usage"]
        status = "ok" if entry["finished"] else ("ERROR" if entry["error"] else "no-final")
        print(f"  [{n}/{len(items)}] {item['id']}: {status} | steps={u['steps']} | "
              f"tokens={u['total_tokens']} | {u['latency_seconds']}s")
    return results

async def run_eval(items: list[dict], runs: int, verbose: bool = False) -> list[dict[str, dict]]:
    """Start the server once, run `runs` passes sequentially, one log file per pass."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    all_runs: list[dict[str, dict]] = []

    async with CtxhClient() as client:
        specs = await client.list_tools()
        for r in range(1, runs + 1):
            log_path = LOG_DIR / f"eval_run{r}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
            print(f"\n=== Lượt {r}/{runs} → {log_path.name} ===")
            all_runs.append(await run_testset(client, specs, items, r, log_path, verbose))

    return all_runs

# Scoring RAGAS

RESULTS_DIR = _HERE / "results"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")
JUDGE_TEMPERATURE = 0.0 # same input gave same score
JUDGE_BASE_URL = "https://api.groq.com/openai/v1"
JUDGE_MAX_TOKENS = 4096 # gpt oss needs a lot of token      
JUDGE_TIMEOUT_S = 120
JUDGE_RETRIES = 2
JUDGE_WORKERS = 1 # how many question is judged at once
RAGAS_METRICS = ("faithfulness", "context_recall")
RAGAS_STATUSES = ("scored", "not_finished", "no_context", "judge_error")

def _ragas_version() -> str | None:
    try:
        import ragas
        return getattr(ragas, "__version__", "unknown")
    except ImportError:
        return None
    
def _make_ragas_judge():
    """
    Build the RAGAS judge (Groq via the OpenAI-compatible endpoint). Imports are lazy so --skip-ragas
    and --selftest don't need ragas installed. Returns a coroutine: judge(sample) -> {metric: float}.
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness, ContextRecall
    client = AsyncOpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url=JUDGE_BASE_URL,
                         timeout=JUDGE_TIMEOUT_S, max_retries=2)
    llm = llm_factory(JUDGE_MODEL, provider="openai", client=client,
                      temperature=JUDGE_TEMPERATURE, max_tokens=JUDGE_MAX_TOKENS)
    faithfulness = Faithfulness(llm=llm)
    recall = ContextRecall(llm=llm)
    async def judge(sample: dict) -> dict:
        f = await faithfulness.ascore(user_input=sample["question"], response=sample["answer"],
                                      retrieved_contexts=sample["contexts"])
        r = await recall.ascore(user_input=sample["question"], retrieved_contexts=sample["contexts"],
                                reference=sample["reference"])
        return {"faithfulness": getattr(f, "value", f), "context_recall": getattr(r, "value", r)}
    return judge

# decide which question to judge:
# unfinished answers -> not fair; answers with no RAGs -> nothing to judge
def ragas_status(res: dict | None) -> str:
    """Pre-scoring label, in this order: not_finished → no_context → ready (eligible for scoring)."""
    if res is None or not res.get("finished") or res.get("error"):
        return "not_finished"
    if not res.get("contexts"):
        return "no_context"
    return "ready"

# build what RAGAS needs: the question, the agent’s answer, 
# the retrieved text, and the reference answer from the testset.
def _sample_of(item: dict, res: dict) -> dict:
    return {"question": item["question"], "answer": res["answer"],
            "contexts": list(res["contexts"]), "reference": item["reference"]}

# like biometric on phone, it hash the sample and the judge's setting
# judging cost a lot of token so a rerun with fingerprint help skipping unchanged questions
# reason why not use question_id is cuz it would reuse old scores 
#   and a hash will ignore stuffs with old fingerprint

def _fingerprint(sample: dict) -> str:
    """Cache key: if the answer, context, reference or judge changes, the old score can no longer be reused."""
    payload = json.dumps({**sample, "judge": JUDGE_MODEL, "temperature": JUDGE_TEMPERATURE},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

def _load_cache(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}

# the judging loop
async def _judge_with_retry(judge, sample: dict, backoff: float) -> dict:
    last: Exception | None = None
    for attempt in range(1, JUDGE_RETRIES + 1):
        try:
            # wait_for each calls until timeout
            scores = await asyncio.wait_for(judge(sample), timeout=JUDGE_TIMEOUT_S)
            out = {}
            for m in RAGAS_METRICS:
                v = scores.get(m)
                # check if score is valid to not corrupt avg since RAGAS sometime returns NaN
                if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v):
                    raise ValueError(f"giám khảo trả về {m} = {v!r}")
                out[m] = round(float(v), 4)
            return out
        except Exception as e: # rate limit, broken json, timeout...
            last = e
            if attempt < JUDGE_RETRIES:
                await asyncio.sleep(backoff * 2 ** (attempt - 1) + random.uniform(0, backoff / 2))
    raise last  # type: ignore[misc]

# Helper to make the same result record every time: status, two scores, error text, 
# and whether it came from the cache. It just for shortening the scoring ragas loop
def _detail(status: str, scores: dict | None = None, error: str | None = None,
            cached: bool = False) -> dict:
    return {"status": status, 
            "faithfulness": (scores or {}).get("faithfulness"),
            "context_recall": (scores or {}).get("context_recall"), 
            "error": error, 
            "cached": cached}

# the ragas scoring loop, the main logic
async def _score_ragas_async(items, results, run_index, judge, rescore, workers, backoff, results_dir):
    cache_path = results_dir / f"ragas_run{run_index}.json"
    old_cache = _load_cache(cache_path)
    usable = {} if rescore else old_cache
    details: dict[str, dict] = {}
    todo: list[tuple[str, dict, str]] = []
    # sort the questionss
    for it in items:
        if not it["ragas"]:
            continue
        res = results.get(it["id"])
        status = ragas_status(res)
        if status != "ready":
            details[it["id"]] = _detail(status)
            continue
        sample = _sample_of(it, res)
        key = _fingerprint(sample)
        hit = usable.get(it["id"])
        if hit and hit.get("key") == key:
            details[it["id"]] = _detail("scored", hit, cached=True)
        else:
            todo.append((it["id"], sample, key))
    new_cache: dict[str, dict] = {}
    if todo:
        setup_error = None
        # build the judge
        try:
            judge_fn = judge if judge is not None else _make_ragas_judge()
        except Exception as e: # no imported ragas/openai, lack key, wrong API...
            judge_fn, setup_error = None, f"không dựng được giám khảo: {type(e).__name__}: {e}"
        # judge the questions asyncly
        sem = asyncio.Semaphore(max(1, workers))
        async def one(qid: str, sample: dict, key: str):
            async with sem:
                if judge_fn is None:
                    return qid, key, None, setup_error
                try: # every questions has their own try/except
                    return qid, key, await _judge_with_retry(judge_fn, sample, backoff), None
                except Exception as e:
                    return qid, key, None, f"{type(e).__name__}: {e}"
        for qid, key, scores, err in await asyncio.gather(*(one(*t) for t in todo)):
            if scores is None:
                details[qid] = _detail("judge_error", error=err)   # error -> no save cache
            else:
                details[qid] = _detail("scored", scores)
                new_cache[qid] = {"key": key, **scores}
    
    # success scores are stored in new cache
    if new_cache: # append cache -> no lost old cache
        results_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({**old_cache, **new_cache}, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    scored = [d for d in details.values() if d["status"] == "scored"]
    return {
        "details": details,
        "counts": {s: sum(1 for d in details.values() if d["status"] == s) for s in RAGAS_STATUSES},
        **{m: _mean([d[m] for d in scored]) for m in RAGAS_METRICS},
    }

# just a wrapper for score_rapas_async so _main can call it
def score_ragas(items: list[dict], results: dict[str, dict], run_index: int, *, judge=None,
                rescore: bool = False, workers: int = JUDGE_WORKERS, backoff: float = 2.0,
                results_dir: Path = RESULTS_DIR) -> dict:
    """Score faithfulness + context_recall for the ragas=true questions of ONE run.
    Per-question label: not_finished (unfinished/has error) → no_context (empty contexts) → scored → judge_error.
    The first two labels are excluded from the sample; judge_error leaves the scores empty and stays out of the average.
    Returns {"details": {id: {...}}, "counts": {label: number of questions}, "faithfulness": avg, "context_recall": avg}.
    `judge` (optional): a coroutine replacing the real judge, used for testing."""
    
    return asyncio.run(_score_ragas_async(items, results, run_index, judge, rescore,
                                          workers, backoff, results_dir))

# csv and summary

CSV_COLUMNS = ["run", "id", "type", "scope", "finished", "error", "ragas_status",
               "calls", "steps", "prompt_tokens", "completion_tokens", "total_tokens",
               "max_prompt", "latency_seconds",
               "faithfulness", "context_recall", "refusal_passed", "refusal_reasons",
               "n_contexts", "n_dense", "n_hybrid", "n_other"]

# arithmetic and statistic calculation helpers
def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None

def _std(xs) -> float | None:
    """Sample standard deviation (divide by n-1). None if fewer than 2 values."""
    xs = [x for x in xs if x is not None]
    return round(statistics.stdev(xs), 4) if len(xs) >= 2 else None

# bundles {mean, std, per_run} for the summary file.
def _stat(per_run: list) -> dict:
    return {"mean": _mean(per_run), "std": _std(per_run), "per_run": per_run}

# count the search_document calls
# this is the evidence that both retrieval modes were actually used.
def count_modes(entry: dict | None) -> dict:
    """Count search_documents calls by mode: dense (including when omitted), hybrid, other (invalid value)."""    
    c = {"dense": 0, "hybrid": 0, "other": 0}
    for s in (entry or {}).get("steps", []):
        if s.get("tool") == "search_documents":
            mode = (s.get("args") or {}).get("mode", "dense")
            c[mode if mode in ("dense", "hybrid") else "other"] += 1
    return c

# read tokens, calls and latency from the log’s usage block
# and falls back to adding up the steps if a field is missing
def _usage_row(entry: dict | None) -> dict:
    """calls, max_prompt... taken from usage if present; otherwise computed from steps."""    
    steps = (entry or {}).get("steps", [])
    u = (entry or {}).get("usage") or {}
    p = u.get("prompt_tokens", sum(s.get("prompt_tokens", 0) for s in steps))
    c = u.get("completion_tokens", sum(s.get("completion_tokens", 0) for s in steps))
    return {
        "calls": u.get("calls", len(steps)),
        "steps": u.get("steps", len(steps)),
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": u.get("total_tokens", p + c),
        "max_prompt": u.get("max_prompt", max((s.get("prompt_tokens", 0) for s in steps), default=0)),
        "latency_seconds": u.get("latency_seconds", round(sum(s.get("latency", 0.0) for s in steps), 2)),
    }

# the actual function that will calculate and save the statistics baase on the functions above
def aggregate(items: list[dict], all_runs: list[dict[str, dict]], refusal_runs: list[dict],
              ragas_runs: list[dict] | None, config: dict) -> tuple[list[dict], dict]:
    """Merge the results of all runs into (per-question CSV rows, summary).
    Average within each run first, then take the mean and standard deviation across runs.
    ragas_runs = None means --skip-ragas."""

    # one CSV row per (run, question), combining the log, the refusal result and the RAGAS result
    # a question with no log gets error = "missing_result" and zeros
    rows: list[dict] = []
    for r, results in enumerate(all_runs, start=1):
        refusal = {d["id"]: d for d in refusal_runs[r - 1]["details"]}
        ragas = ragas_runs[r - 1]["details"] if ragas_runs else {}
        for it in items:
            e = results.get(it["id"])
            ref, rg = refusal.get(it["id"]), ragas.get(it["id"])
            modes = count_modes(e)
            rows.append({
                "run": r, "id": it["id"], "type": it["type"], "scope": it.get("scope", ""),
                "finished": bool(e and e.get("finished")),
                "error": (e or {}).get("error") or ("" if e else "missing_result"),
                "ragas_status": (rg["status"] if rg else ("skipped" if it["ragas"] and ragas_runs is None else "")),
                **_usage_row(e),
                "faithfulness": rg["faithfulness"] if rg else None,
                "context_recall": rg["context_recall"] if rg else None,
                "refusal_passed": ref["passed"] if ref else None,
                "refusal_reasons": ";".join(ref["reasons"]) if ref else "",
                "n_contexts": len((e or {}).get("contexts", [])),
                "n_dense": modes["dense"], "n_hybrid": modes["hybrid"], "n_other": modes["other"],
            })
    n_runs = len(all_runs)
    present = [r for r in rows if r["error"] != "missing_result"]
    scopes = sorted({s for rr in refusal_runs for s in rr["by_scope"]})
    # averages within each run first, then takes the mean and standard deviation across runs
    summary = {
        "config": config,
        "ragas": {"skipped": True} if ragas_runs is None else {
            "skipped": False,
            **{m: _stat([rr[m] for rr in ragas_runs]) for m in RAGAS_METRICS},
            "status_counts": {
                **{s: sum(rr["counts"][s] for rr in ragas_runs) for s in RAGAS_STATUSES},
                "per_run": [rr["counts"] for rr in ragas_runs],
            },
        },
        "refusal": {
            "overall": _stat([rr["rate"] for rr in refusal_runs]),
            "by_scope": {s: _stat([rr["by_scope"].get(s, {}).get("rate") for rr in refusal_runs])
                         for s in scopes},
            "per_run": [{"run": i, "passed": rr["passed"], "total": rr["total"], "rate": rr["rate"],
                         "by_scope": rr["by_scope"]} for i, rr in enumerate(refusal_runs, start=1)],
        },
        "modes": {
            **{k: sum(r[f"n_{k}"] for r in rows) for k in ("dense", "hybrid", "other")},
            "per_run": [{k: sum(r[f"n_{k}"] for r in rows if r["run"] == i)
                         for k in ("dense", "hybrid", "other")} for i in range(1, n_runs + 1)],
        },
        "usage": {      # only count questions that actually have a log (missing results don't pull the average toward 0)
            "n_question_runs": len(present),
            "mean_total_tokens": _mean([r["total_tokens"] for r in present]),
            "mean_prompt_tokens": _mean([r["prompt_tokens"] for r in present]),
            "mean_completion_tokens": _mean([r["completion_tokens"] for r in present]),
            "mean_latency_seconds": _mean([r["latency_seconds"] for r in present]),
            "mean_calls": _mean([r["calls"] for r in present]),
            "max_prompt_max": max((r["max_prompt"] for r in present), default=0),
        },
        "per_question": {},
    }
    for it in items:
        mine = [r for r in rows if r["id"] == it["id"]]
        summary["per_question"][it["id"]] = {
            "faithfulness": _mean([r["faithfulness"] for r in mine]),
            "context_recall": _mean([r["context_recall"] for r in mine]),
            "refusal_pass_rate": _mean([1.0 if r["refusal_passed"] else 0.0
                                        for r in mine if r["refusal_passed"] is not None]),
        }
    return rows, summary

# write the CSV with encoding="utf-8-sig", so Excel shows the vnese correctly 
# and the summary as JSON with ensure_ascii=False so vnese isn’t stored as \u1ebf codes.
def write_outputs(rows: list[dict], summary: dict, results_dir: Path = RESULTS_DIR) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = results_dir / "eval_per_question.csv", results_dir / "eval_summary.json"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:      # utf-8-sig: Excel reads Vietnamese correctly
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in CSV_COLUMNS})
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path

# _fmt and print_summary produce the short table: 0.912 ± 0.020 for scores, 
# n/a when there’s no data, and the file paths at the end.
def _fmt(stat: dict | None) -> str:
    if not stat or stat.get("mean") is None:
        return "n/a"
    return f"{stat['mean']:.3f}" + (f" ± {stat['std']:.3f}" if stat.get("std") is not None else "")

def print_summary(summary: dict, csv_path: Path, json_path: Path) -> None:
    c, rg, rf = summary["config"], summary["ragas"], summary["refusal"]
    m, u = summary["modes"], summary["usage"]
    print(f"\n{'=' * 64}\nKẾT QUẢ EVAL: {c['runs']} lượt × {c['n_questions']} câu\n{'=' * 64}")
    if rg["skipped"]:
        print("RAGAS                  (bỏ qua: --skip-ragas)")
    else:
        sc = rg["status_counts"]
        print(f"RAGAS faithfulness     {_fmt(rg['faithfulness'])}")
        print(f"RAGAS context_recall   {_fmt(rg['context_recall'])}")
        print(f"  số câu: scored={sc['scored']} not_finished={sc['not_finished']} "
              f"no_context={sc['no_context']} judge_error={sc['judge_error']}")
    print(f"Từ chối đúng (tổng)    {_fmt(rf['overall'])}")
    for scope, st in rf["by_scope"].items():
        print(f"  {scope:<20} {_fmt(st)}")
    print(f"Truy vấn               dense={m['dense']}  hybrid={m['hybrid']}  other={m['other']}")
    print(f"Mỗi câu (trung bình)   {u['mean_total_tokens']} token | {u['mean_latency_seconds']}s | "
          f"{u['mean_calls']} lượt gọi LLM | max_prompt lớn nhất = {u['max_prompt_max']}")
    print(f"File kết quả           {csv_path}\n                       {json_path}")

# Read reused logs
def latest_log(run_index: int) -> Path | None:
    files = sorted(LOG_DIR.glob(f"eval_run{run_index}_*.jsonl"))
    return files[-1] if files else None

def load_logs(runs: int) -> tuple[list[dict[str, dict]], list[Path]]:
    """Read the latest eval log of each run into {question id: log line}, without re-running the agent."""
    all_runs, paths = [], []
    for r in range(1, runs + 1):
        path = latest_log(r)
        if path is None:
            raise FileNotFoundError(f"không có log eval_run{r}_*.jsonl trong {LOG_DIR}; hãy chạy eval không có --reuse-logs trước")
        entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        all_runs.append({e["id"]: e for e in entries})
        paths.append(path)
    return all_runs, paths

# Self-test checkpoints

def _selftest_refusal() -> None:
    """3 fake answers: correct refusal / fabricated / ran out of MAX_STEPS. Expected: pass, fail, fail."""
    items = [
        {"id": "N1", "type": "no_answer", "scope": "in_domain",
         "expected": {"refusal_markers": ["không đề cập", "không có quy định"]}},
        {"id": "N2", "type": "no_answer", "scope": "in_domain",
         "expected": {"refusal_markers": ["không đề cập", "không có quy định"]}},
        {"id": "N3", "type": "no_answer", "scope": "out_of_domain",
         "expected": {"refusal_markers": ["không đề cập", "không có quy định"]}},
    ]
    search = [{"tool": "search_documents"}]
    results = {
        "N1": {"answer": "QĐ 224 KHÔNG ĐỀ CẬP mức phạt này.", "finished": True, "steps": search},
        "N2": {"answer": "Mức phạt là 500.000 đồng.", "finished": True, "steps": search},
        "N3": {"answer": "Xin lỗi, mình chưa tìm được câu trả lời.", "finished": False, "steps": search},
    }
    out = score_refusal(items, results)
    got = [(d["id"], d["passed"], d["reasons"]) for d in out["details"]]
    print("selftest:", got)
    assert got[0][1] is True
    assert got[1][1] is False and got[1][2] == ["no_refusal_marker"]
    assert got[2][1] is False and "not_finished" in got[2][2]
    print("selftest OK")

def _selftest_ragas() -> None:
    """Status labels + cache + retry of score_ragas, using a fake judge (no API calls)."""    
    import tempfile
    calls: Counter = Counter()
    async def fake_judge(sample: dict) -> dict:
        calls[sample["question"]] += 1
        if sample["question"] == "D":
            raise RuntimeError("429 rate limit")
        return {"faithfulness": 0.8, "context_recall": 0.6}
    def res(answer="a", finished=True, contexts=("c",), error=None):
        return {"answer": answer, "finished": finished, "error": error, "contexts": list(contexts), "steps": []}
    items = [{"id": k, "question": k, "reference": "ref", "ragas": True} for k in "ABCD"]
    items.append({"id": "E", "question": "E", "reference": None, "ragas": False})
    results = {"A": res(), "B": res(finished=False), "C": res(contexts=[]), "D": res(), "E": res()}
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(judge=fake_judge, backoff=0.0, results_dir=Path(tmp))
        out = score_ragas(items, results, 1, **kw)
        st = {k: v["status"] for k, v in out["details"].items()}
        print("selftest ragas:", st, out["counts"])
        assert st == {"A": "scored", "B": "not_finished", "C": "no_context", "D": "judge_error"}, st
        assert out["counts"] == {"scored": 1, "not_finished": 1, "no_context": 1, "judge_error": 1}
        assert out["faithfulness"] == 0.8 and out["context_recall"] == 0.6   # only count scored
        assert calls["A"] == 1 and calls["D"] == JUDGE_RETRIES and "E" not in calls
        score_ragas(items, results, 1, **kw) # cache: A -> no recall, D -> recall
        assert calls["A"] == 1 and calls["D"] == 2 * JUDGE_RETRIES
        score_ragas(items, results, 1, rescore=True, **kw) # --rescore: force scoring again
        assert calls["A"] == 2
        results["A"] = res(answer="câu trả lời khác") # changed answer -> abandon old cache
        score_ragas(items, results, 1, **kw)
        assert calls["A"] == 3
        cached = json.loads((Path(tmp) / "ragas_run1.json").read_text(encoding="utf-8"))
        assert list(cached) == ["A"] # judge_error is not cache-d
    print("selftest ragas OK")

def _selftest_aggregate() -> None:
    """Aggregated figures with 3 fake runs whose results can be computed by hand."""
    import tempfile
    items = [
        {"id": "Q1", "type": "answerable", "ragas": True, "question": "q1", "reference": "r"},
        {"id": "Q2", "type": "no_answer", "ragas": False, "scope": "in_domain", "question": "q2",
         "reference": None, "expected": {"refusal_markers": ["không đề cập"]}},
    ]
    search = lambda **a: {"tool": "search_documents", "args": a, "prompt_tokens": 10}
    runs, refusal_runs, ragas_runs = [], [], []
    for r, (faith, tokens, answer) in enumerate([(0.5, 100, "Không đề cập."), (0.7, 200, "Không đề cập."),
                                                 (0.9, 300, "Mức phạt 500k.")], start=1):
        res = {
            "Q1": {"answer": "x", "finished": True, "error": None, "contexts": ["c1", "c2"],
                   "usage": {"calls": 2, "steps": 2, "prompt_tokens": tokens - 10, "completion_tokens": 10,
                             "total_tokens": tokens, "max_prompt": 60, "latency_seconds": 1.5},
                   "steps": [search(query="a"), search(query="b", mode="hybrid"), search(query="c", mode="fuzzy")]},
            "Q2": {"answer": answer, "finished": True, "error": None, "contexts": [],
                   "steps": [search(query="d", mode="dense")]}, # no usage -> calculate from steps
        }
        runs.append(res)
        refusal_runs.append(score_refusal(items, res))
        ragas_runs.append({"details": {"Q1": _detail("scored", {"faithfulness": faith, "context_recall": 1.0})},
                           "counts": {"scored": 1, "not_finished": 0, "no_context": 0, "judge_error": 0},
                           "faithfulness": faith, "context_recall": 1.0})
    rows, s = aggregate(items, runs, refusal_runs, ragas_runs, {"runs": 3, "n_questions": 2})
    print("selftest aggregate:", s["ragas"]["faithfulness"], s["refusal"]["overall"], s["modes"])
    assert len(rows) == 6
    assert s["ragas"]["faithfulness"]["mean"] == 0.7 and s["ragas"]["faithfulness"]["std"] == 0.2
    assert s["refusal"]["overall"]["per_run"] == [1.0, 1.0, 0.0]
    assert s["refusal"]["overall"]["mean"] == 0.6667 and s["refusal"]["overall"]["std"] == 0.5774
    assert s["refusal"]["by_scope"]["in_domain"]["mean"] == 0.6667
    assert (s["modes"]["dense"], s["modes"]["hybrid"], s["modes"]["other"]) == (6, 3, 3)
    q2 = [r for r in rows if r["id"] == "Q2"][0]
    assert q2["calls"] == 1 and q2["max_prompt"] == 10 # generate from steps since no usage
    assert s["usage"]["max_prompt_max"] == 60
    assert s["per_question"]["Q1"]["faithfulness"] == 0.7 and s["per_question"]["Q2"]["refusal_pass_rate"] == 0.6667
    with tempfile.TemporaryDirectory() as tmp:
        csv_path, json_path = write_outputs(rows, s, Path(tmp))
        assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf") # utf-8-sig for vnese
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            back = list(csv.DictReader(f))
        assert list(back[0]) == CSV_COLUMNS and len(back) == 6
        assert json.loads(json_path.read_text(encoding="utf-8"))["config"]["runs"] == 3
    print("selftest aggregate OK")

def _selftest() -> None:
    _selftest_refusal()
    _selftest_ragas()
    _selftest_aggregate()

def main() -> None:
    parser = argparse.ArgumentParser(description="Eval Trợ lý CTXH")
    parser.add_argument("--selftest", action="store_true", help="chạy các mốc kiểm tra offline rồi thoát")
    parser.add_argument("--runs", type=int, default=1, help="số lượt chạy testset (đề cần ≥3 log thật)")
    parser.add_argument("--limit", type=int, default=None, help="chỉ chạy N câu đầu (để kiểm tra nhanh)")
    parser.add_argument("--verbose", action="store_true", help="in từng bước của agent ra stderr")
    parser.add_argument("--skip-ragas", action="store_true", help="bỏ qua bước chấm RAGAS (chạy thử nhanh)")
    parser.add_argument("--rescore", action="store_true", help="ép chấm RAGAS lại, bỏ qua cache results/ragas_run<n>.json")
    parser.add_argument("--workers", type=int, default=JUDGE_WORKERS, help="số câu chấm RAGAS song song")
    parser.add_argument("--reuse-logs", action="store_true",
                        help="không chạy lại agent: đọc log eval_run<n>_*.jsonl mới nhất rồi chấm + tổng hợp")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
        return
    
    started = datetime.now()
    items = load_testset()
    summarize_testset(items)
    if args.limit:
        items = items[:args.limit]
        print(f"(chỉ chạy {len(items)} câu đầu theo --limit)")

    # Start agent or re-read old logs
    if args.reuse_logs:
        all_runs, log_paths = load_logs(args.runs)
        print(f"Dùng lại log: {[p.name for p in log_paths]}")
    else:
        all_runs = asyncio.run(run_eval(items, args.runs, args.verbose))
        log_paths = [latest_log(r) for r in range(1, args.runs + 1)]

    # score refused run and ragas accordingly
    refusal_runs = [score_refusal(items, results) for results in all_runs]
    ragas_runs = None
    if not args.skip_ragas:
        ragas_runs = []
        for r, results in enumerate(all_runs, start=1):
            print(f"\nChấm RAGAS lượt {r}/{len(all_runs)} ...")
            ragas_runs.append(score_ragas(items, results, r, rescore=args.rescore, workers=args.workers))
            errs = sorted({d["error"] for d in ragas_runs[-1]["details"].values() if d["status"] == "judge_error"})
            if errs: # show the actual cause and not just judge_error score
                print(f"  judge_error ({ragas_runs[-1]['counts']['judge_error']} câu), lỗi đầu tiên: {errs[0][:300]}")

    # summary and csv
    models = sorted({e["model"] for results in all_runs for e in results.values() if e.get("model")})
    config = {
        "agent_model": models[0] if len(models) == 1 else models,
        "judge_model": None if args.skip_ragas else JUDGE_MODEL,
        "judge_temperature": None if args.skip_ragas else JUDGE_TEMPERATURE,
        "ragas_version": None if args.skip_ragas else _ragas_version(),
        "runs": len(all_runs),
        "n_questions": len(items),
        "n_ragas_questions": sum(1 for it in items if it["ragas"]),
        "std": "sample (n-1)",
        "reuse_logs": args.reuse_logs,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "log_files": [str(p) for p in log_paths],
    }
    rows, summary = aggregate(items, all_runs, refusal_runs, ragas_runs, config)
    csv_path, json_path = write_outputs(rows, summary)
    print_summary(summary, csv_path, json_path)

if __name__ == "__main__":
    main()