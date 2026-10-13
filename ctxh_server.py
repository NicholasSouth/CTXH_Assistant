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

# Helpers for the 7 functions:
# get_document, list_documents, find_activity, 
# convert_points_to_days, list_activities, register_activity, list_registrations
import math
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime
CODES_PATH = _HERE / 'data' / 'activity_codes.json'
ACTIVITIES_PATH = _HERE / 'data' / 'activities.json'
REGISTRATIONS_PATH = _HERE / 'resources' / 'registrations.json'
# Use web designing rules to make analogy for the helpers, then:
# The stuffs in /data and /resources are the database, sql stuffs
# ActivityCode, Activity, and Registration are the models for the database
# _read_json(path), _load_codes(),
# _load_activities(), _load_registrations() 
#   are the DAOs in charge of database-project communication
# The rest are Services - logic handling stuffs
@dataclass
class ActivityCode: # model for /data/activity_codes.json
    code: str
    field: str
    description: str
    points_min: float
    points_max: float
    days_min: float
    days_max: float
@dataclass
class Activity: # model for /data/activities.json
    activity_id: str
    name: str
    code: str
    date: str
    organizer: str
    capacity: int
    status: str
# This model is a student's sign up record
# The registration record should be stored in REGISTRATIONS_PATH like declared above
@dataclass
class Registration: 
    registration_id: str
    student_id: str
    activity_id: str
    created_at: str
# basic json reader code
# Registrations change at runtime so we read it a lot, better make a fucntion for fast using
def _read_json(path: Path): 
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
# utilize _read_json to load data from /data/activity_codes.json into 
#   a list of ActivityCode objects that Python can handle
def _load_codes() -> list[ActivityCode]:
    return [ActivityCode(**row) for row in _read_json(CODES_PATH)]
# use _read_json to to the same from /data/activities.json to a list of Activity objects
def _load_activities() -> list[Activity]:
    return [Activity(**row) for row in _read_json(ACTIVITIES_PATH)]
# do the same from REGISTRATIONS_PATH to a Registration object
def _load_registrations() -> list[Registration]:
    # since registration records could be empty, we also have to check its exsistence
    if not REGISTRATIONS_PATH.exists():
        return []
    raw = _read_json(REGISTRATIONS_PATH)
    # incase the file contain only a single dict, we wrap it in a list
    if isinstance(raw, dict):
        raw = [raw]
    return [Registration(**row) for row in raw]
# save a Registration list into registrations.json
def _save_registrations(regs: list[Registration]) -> None:
    # check the exsistence of the file
    REGISTRATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # write the list into a temporary json incase somthing wrong half way
    tmp = REGISTRATIONS_PATH.with_suffix('.json.tmp')
    tmp.write_text(
        json.dumps([asdict(r) for r in regs], ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    # after check, swap the two file, ensuring no half written content
    os.replace(tmp, REGISTRATIONS_PATH)
# clean text for searching
def _fold(text: str) -> str:
    # lowercase + strip Vietnamese accents, Ex: 'Hiến máu' -> 'hien mau'
    # since user might search without accent and cases
    text = text.replace('đ', 'd').replace('Đ', 'D')
    decomposed = unicodedata.normalize('NFD', text)
    return ''.join(c for c in decomposed if unicodedata.category(c) != 'Mn').lower()
# clean floating point numbers into string
def _num(x: float) -> str: 
    return str(int(x)) if float(x).is_integer() else str(x)   # Ex:  2.0 -> '2', 2.5 -> '2.5'
# clean an interval
# Ex: [5; 5] -> show 5; if [5; 10] -> show 5 - 10
def _range(lo: float, hi: float, unit: str) -> str: 
    return f'{_num(lo)} {unit}' if lo == hi else f'{_num(lo)}-{_num(hi)} {unit}'
# calculate the remaining slot for an activity
def _slots_left(activity: Activity, regs: list[Registration]) -> int:
    # loop through list of registrations and count the registrations with the matching activity_id 
    taken = sum(1 for r in regs if r.activity_id == activity.activity_id)
    # use max incase event overbook -> remaining is negative -> use max to show 0 instead
    return max(0, activity.capacity - taken)
#_____ Helpers list end.

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
    6000 characters, with '[... truncated ...]' added at the end. If the doc_id
    does not exist, returns "No document with id '<doc_id>'."
    """

    # This server is pretty much like get_document in vector_store
    doc_id = doc_id.strip() #strip incase whitespace characters
    # Just like a servlet, catching request attributes and call services
    #   the tool also catch the parameters sent by the agent, calling other functions
    #   and return str back so the agent understand
    chunks = STORE.get_chunks_by_doc(doc_id) 
    if not chunks:
        return f"No document with id '{doc_id}'."
    # sort the chunks in order too, make it easier for the agent to understand the context
    chunks.sort(key=lambda c: c.chunk_index)
    text = '\n'.join(c.content for c in chunks)
    # cap the text at 6000 (original 400) incase overloading
    if len(text) > 6000:
        text = text[:6000] + '\n[... truncated ...]'
    return text

@mcp.tool()
def list_documents() -> str:
    """
    List every regulation document, one per line, as 'doc_id: title'.
    Use it to see what topics the corpus covers or to look up a doc_id. Takes no arguments.
    """

    # same framework in get_doc, this server is pretty much the same as that in vector_store
    docs = STORE.list_all()
    if not docs:
        return 'No documents in the corpus'
    return '\n'.join(f'{d.doc_id}: {d.title}' for d in docs)

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

    q = query.strip() # at this point, strip every input str incase whitespace, pretty much self explainatory
    if not q:
        return 'Error: query is empty.'
    try:
        codes = _load_codes() # load the activity code list for later checking
    # check error incase it stops the server, instead it should log this in and keep server running
    #   OSError for missing/deleted/not permitted files
    #   ValueError for invalid values in file, TypeError for invalid file type
    except (OSError, ValueError, TypeError) as error: 
        log.exception('find_activity failed') # log the error into server.log
        return f'Error: cannot read activity code table ({error}).'
    # split each words in the query
    # check if any individual word is purely digit then compare to the activity_id in codes
    digits = [w for w in q.split() if w.isdigit()]
    if digits:
        matches = [c for c in codes if c.code.startswith(digits[0])]
    else:
    # check if EVERY SINGLE WORDS of query appear in the field/description of all activities in codes
    # ofc _fold every words for easier comparision
        words = _fold(q).split()
        matches = [c for c in codes
                   if all(w in _fold(c.field + ' ' + c.description) for w in words)]
    if not matches:
        return 'No matching activity'
    return '\n'.join(
        f'{c.code} | {c.field} | {c.description} | '
        f'{_range(c.points_min, c.points_max, "điểm")} | '
        f'{_range(c.days_min, c.days_max, "ngày")}'
        for c in matches[:5]
    )

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

    # validate a point
    if (isinstance(total_points, bool) # Python treats bool as 0 and 1 so we should spite it
            or not isinstance(total_points, (int, float)) # spite float to cuz nums of days cant be float
            or not math.isfinite(total_points)): #ofc spite infinity too
        return 'Error: total_points must be a number.'
    if total_points < 0: # nums of days shouldnt be neg -> spit neg too
        return 'Error: total_points must not be negative.'
    # split points into 2 part, /10 and %10
    # each 10 points is a day -> /10 = days
    # each %10 will be round, just like the rules in the metadata
    whole, rest = divmod(total_points, 10)
    if rest < 5:
        days = whole
    elif rest == 5:
        days = whole + 0.5
    else:
        days = whole + 1
    return _num(days)

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

    # just like in find_activities, we have to check for errors to put into log incase it stops server
    # this we have to consider open status too for students to participate
    try:
        activities = [a for a in _load_activities() if a.status == 'open']
        regs = _load_registrations() #also log the registration too
    except (OSError, ValueError, TypeError) as error:
        log.exception('list_activities failed')
        return f'Error: cannot read activity data ({error}).'
    # implement a smart keyword search
    # but actually its just like in find_activity, _fold the query and search in the activities's name/code
    # except its a leanier condition, doesnt force every words to be inside the name/code
    kw = _fold(keyword.strip())
    if kw:
        activities = [a for a in activities if kw in _fold(a.name) or kw in a.code]
    if not activities:
        return 'No open activities'
    return '\n'.join(
        f'{a.activity_id} | {a.name} | {a.code} | {a.date} | {a.organizer} | '
        f'{_slots_left(a, regs)}'
        for a in activities
    )

# prolly the most important tool in the whole project
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
    
    sid = student_id.strip()
    aid = activity_id.strip().upper()
    # validate ids
    if not sid: # register must have student id
        return 'Error: student_id is empty. Ask the user for their student ID.'
    try:
        # strictly search for the correct activity_id
        activity = next((a for a in _load_activities() if a.activity_id == aid), None)
        # activity_id is a must and the activity must be open to be assigned to a student too
        if activity is None:
            return f'Error: activity {aid} does not exist.'
        if activity.status != 'open':
            return f'Error: activity {aid} is {activity.status}, not open for registration.'
        # mimic _save_registrations, load a temporary list first then append this reg incase error
        # later on we will swap the two list
        regs = _load_registrations()
        # validate reg, this part check any faulty logic when assigning student to activity
        if any(r.student_id == sid and r.activity_id == aid for r in regs):
            return f'Error: student {sid} is already registered for {aid}.'
        if _slots_left(activity, regs) == 0:
            return f'Error: activity {aid} has no slots left.'
        # strip and take the reg_id with highest number to prepare making this reg's id
        last = max((int(r.registration_id[1:]) for r in regs), default=0)
        new = Registration(
            registration_id=f'R{last + 1:04d}',
            student_id=sid,
            activity_id=aid,
            created_at=datetime.now().isoformat(timespec='seconds'),
        )
        regs.append(new)
        _save_registrations(regs)
    # check ing error and logging as usual here
    except (OSError, ValueError, TypeError, KeyError) as error:
        log.exception('register_activity failed')
        return f'Error: registration failed ({error}).'
    log.info('registered %s for %s as %s', sid, aid, new.registration_id)
    return (f'Registered student {sid} for {aid} ({activity.name}). '
            f'registration_id={new.registration_id}')

@mcp.tool()
def list_registrations(student_id: str) -> str:
    """
    List all social-work (CTXH) activities a student has registered for, one per line:
    'registration_id | activity_id | name | code | date'. Read-only.
    
    Returns 'No registrations' if the student has none. Use it to check the 
    student's current registrations, and always use it right after register_activity to verify 
    that the registration really exists.
    """

    sid = student_id.strip()
    if not sid:
        return 'Error: student_id is empty.'
    try:
        # load all registrations which student_id match this into list mine
        mine = [r for r in _load_registrations() if r.student_id == sid]
        # also store the activity_id of the reg for listing the student's activity and stuffs
        by_id = {a.activity_id: a for a in _load_activities()}
    except (OSError, ValueError, TypeError) as error:
        log.exception('list_registrations failed')
        return f'Error: cannot read registrations ({error}).'
    if not mine:
        return 'No registrations'
    lines = []
    # sort from oldest to newest reg (cuz reg_id kept incrementing for newer regs)
    for r in sorted(mine, key=lambda r: r.registration_id):
        a = by_id.get(r.activity_id)
        name, code, date = (a.name, a.code, a.date) if a else ('(unknown activity)', '-', '-')
        lines.append(f'{r.registration_id} | {r.activity_id} | {name} | {code} | {date}')
    return '\n'.join(lines)

if __name__ == "__main__":
    mcp.run(transport="stdio")