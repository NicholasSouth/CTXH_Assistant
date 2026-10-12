import logging
import json
import os
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
from mcp.server import MCPServer
from vector_store import VectorStore
from ex2_1_hybrid_search import build_bm25_index, hybrid_search

LOG_PATH = _HERE/"server.log"
_handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
try:
    _handlers.append(logging.FileHandler(LOG_PATH, encoding="utf-8"))
except OSError as error:
    print(f"warning: cannot open {LOG_PATH}: {error}", file=sys.stderr)
logging.basicConfig(
    level=logging.INFO,
    handlers=_handlers,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

mcp = MCPServer("ctxh-assistant")

INDEX_PATH = Path(os.getenv("RAG_INDEX_PATH", str(_HERE/"data"/"index.json")))
STORE = VectorStore()
STORE.load(str(INDEX_PATH))
CHUNKS = list(STORE.chunks.values())
BM25_INDEX = build_bm25_index(CHUNKS)
EMBEDDINGS = [c.embedding for c in CHUNKS]

@mcp.tool()
def search_documents(query: str, top_k: int = 3, mode: str = "dense") -> str:
    """
    Search the social-work (CTXH, công tác xã hội) regulation documents.

    The documents are in Vietnamese. Write the query in Vietnamese, using the
    user's own words; do not translate it to English.

    mode: 'dense' (semantic only, default) or 'hybrid' (keyword + semantic). Use
    'hybrid' when the query contains an activity code (e.g., '001201') or an exact
    term; use 'dense' for questions in natural language. Any other value returns
    'Error: ...'.

    Returns up to top_k passages (default 3, max 8), most relevant first, one block
    each, formatted as 'doc_id: passage text'. Passages are short and may be cut
    off; call get_document with the doc_id for the full text. If nothing is found,
    returns 'No relevant passages found in the corpus'.
    Use this first for any question about the rules.
    """

    top_k = max(1, min(top_k, 8))
    if mode == "dense":
        results = STORE.search(query, top_k=top_k)
    elif mode == "hybrid":
        ranked = hybrid_search(query, CHUNKS, BM25_INDEX, EMBEDDINGS, top_k=top_k)
        results = [(CHUNKS[i], score) for i, score in ranked]
    else:
        return f"Error: unknown mode '{mode}'. Use 'dense' or 'hybrid'."
    if not results:
        return "No relevant passages found in the corpus"
    return "\n\n".join(f"{c.doc_id}: {c.content}" for c, _s in results)

@mcp.tool()
def get_document(doc_id: str) -> str:
    """
    Read the text of one regulation document by its doc_id, taken from a
    search_documents or list_documents result. Use it when a passage is cut off
    or you need the surrounding context.

    The text is rebuilt by joining the document's chunks in order, so a few
    sentences may appear twice where chunks overlap. Output is truncated after
    4000 characters, with '[... truncated ...]' added at the end. If the doc_id
    does not exist, returns "No document with id '<doc_id>'."
    """


@mcp.tool()
def list_documents() -> str:
    """
    List every regulation document, one per line, as 'doc_id: title'.
    Use it to see what topics the corpus covers or to look up a doc_id. Takes no arguments.
    """

@mcp.tool()
def find_activity(query: str) -> str:
    """
    Find social-work (CTXH) activity codes in the code table by a description or by a code.
    Query can be Vietnamese words (e.g., 'hiến máu', 'dạy kèm') or a code
    or code prefix (e.g., '001201', '005'). Keep the query in Vietnamese; do not
    translate it. Returns up to 5 matches, one per line:
    'code | field | description | points range | days range'.

    The points and days are ranges; the exact value is decided by the organizing unit,
    so never present a single number as final. Returns 'No matching activity' if nothing matches.
    Use it to answer 'how many days is activity X worth'.
    """

@mcp.tool()
def convert_points_to_days(total_points: float) -> str:
    """
    Convert accumulated CTXH (social-work) points into social-work days using the regulation rule
    (section II.2c): divide the total points by 10; the leftover points below
    5 are dropped, exactly 5 leftover points count as half a day, and leftover
    points above 5 round up to one full day.
    Example: 24 points -> 2 days, 25 points -> 2.5 days, 27 points -> 3 days.
    
    Returns the number of days as a plain number. Returns 'Error: ...' if
    total_points is negative or not a number. Use this tool for the arithmetic;
    do not calculate it yourself.
    """

@mcp.tool()
def list_activities(keyword: str = "") -> str:
    """
    List social-work (CTXH) activities that are currently open for registration. Each line:
    'activity_id | name | code | date | organizer | slots left'.
    
    keyword (optional) filters by a word in the name or by the code. It must be in
    Vietnamese, as in the activity names (e.g., 'hiến máu'); never translate it to
    English. Without a keyword, returns all open activities. Use it to find an
    activity_id before calling register_activity. Returns 'No open activities' if
    none match.
    """

@mcp.tool()
def register_activity(student_id: str, activity_id: str) -> str:
    """
    Register a student for a social-work (CTXH) activity. This changes state: it
    writes a new registration record.

    student_id is the student ID given by the user. If the user has not given it,
    ask for it; never invent one. activity_id comes from list_activities.

    Fails with an 'Error: ...' message if the activity does not exist, is closed,
    has no slots left, the student is already registered or student_id is empty.
    On success returns a confirmation with the registration id.

    The confirmation alone is not proof: after calling this tool you must call
    list_registrations for the same student_id and check that the new record
    appears before telling the user the registration succeeded.
    """

@mcp.tool()
def list_registrations(student_id: str) -> str:
    """
    List all social-work (CTXH) activities a student has registered for, one per line:
    'registration_id | activity_id | name | code | date'. Read-only.
    
    Returns 'No registrations' if the student has none. Use it to check the 
    student's current registrations, and always use it right after register_activity to verify 
    that the registration really exists.
    """

if __name__ == "__main__":
    mcp.run(transport="stdio")