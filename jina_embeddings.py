"""
Jina embeddings helper with an OpenAI-like calling convention.

Usage:
    from jina_embeddings import jina_embeddings

    resp = jina_embeddings(
        input=["A beautiful sunset over the beach", "Un beau coucher de soleil sur la plage"],
        api_key=os.getenv("JINA_API_KEY"),
    )
    for item in resp["data"]:
        print(item["embedding"])
"""

import requests

class AttrDict(dict):
    """A dict that also supports attribute access: d.data == d["data"]."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __dir__(self):
        return list(super().__dir__()) + list(self.keys())


def _wrap(obj):
    """Recursively convert dicts to AttrDict so dot access works at every level."""
    if isinstance(obj, dict):
        return AttrDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def jina_embeddings(
    input,
    api_key,
    model: str = "jina-embeddings-v3",
    task: str = "retrieval.passage",
    normalized: bool = True,
    dimensions: int | None = None,
    timeout: int = 30,
):
    """
    Get embeddings from the Jina AI embeddings API, mirroring the shape of
    OpenAI's `client.embeddings.create(...)` response.

    Parameters
    ----------
    input : str or list[str]
        A single string or a list of strings to embed. (Matches OpenAI's
        `input` parameter, which also accepts either form.)
    api_key : str
        Your Jina API key.
    model : str
        Jina embedding model name. Default: "jina-embeddings-v3".
    task : str
        Jina-specific task hint that optimizes the embedding for its use case.
        Common values: "retrieval.query", "retrieval.passage", "text-matching",
        "classification", "separation". Default: "retrieval.passage"
        (use "retrieval.query" when embedding search queries).
    normalized : bool
        Whether to L2-normalize the returned vectors. Default: True.
    dimensions : int, optional
        Truncate output embeddings to this many dimensions, if the model
        supports Matryoshka truncation (jina-embeddings-v3 does).
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    AttrDict (a dict subclass)
        A response shaped like OpenAI's embeddings response. Supports both
        bracket access (`resp["data"][0]["embedding"]`) and attribute access
        (`resp.data[0].embedding`), recursively at every level:
        {
            "data": [
                {"index": 0, "embedding": [...]},
                {"index": 1, "embedding": [...]},
                ...
            ],
            "model": "jina-embeddings-v3",
            "usage": {...},
        }
        So you can call `resp["data"][0]["embedding"]` just like
        `resp.data[0].embedding` in the OpenAI SDK.

    Raises
    ------
    requests.HTTPError
        If the Jina API returns a non-2xx response.
    ValueError
        If api_key is missing.
    """
    if not api_key:
        raise ValueError("api_key is required (e.g. os.getenv('JINA_API_KEY'))")

    # Normalize input to a list, same way OpenAI's client accepts str or list[str]
    if isinstance(input, str):
        texts = [input]
    else:
        texts = list(input)

    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "task": task,
        "normalized": normalized,
        "input": [{"text": t} for t in texts],
    }
    if dimensions is not None:
        payload["dimensions"] = dimensions

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    result = resp.json()

    # Jina already returns {"data": [{"index":..., "embedding":...}, ...], "model":..., "usage":...}
    # which is already OpenAI-shaped, but we normalize defensively in case of
    # missing "index" keys so downstream code can rely on order + index.
    data = result.get("data", [])
    for i, item in enumerate(data):
        item.setdefault("index", i)

    return _wrap({
        "data": data,
        "model": result.get("model", model),
        "usage": result.get("usage", {}),
    })