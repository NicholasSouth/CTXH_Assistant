# AI Usage

- **Tool:** Claude

## Part that uses Claude

### 1. Parsing and validating the model's replies

**Prompt:**

```text
Tôi đang viết một agent ReAct bằng Python (không dùng framework agent). Mô hình
sẽ trả về văn bản thô theo một trong hai dạng:
Thought: ...
Action: <tên_tool> <JSON tham số>
hoặc
Thought: ...
Final Answer: ...
Hãy viết hai thành phần sau.

1. Dataclass `Parsed` và hàm `parse_reply(reply: str) -> Parsed`.
   - `Parsed.kind` là "action", "final" hoặc "error".
   - Cắt bỏ mọi thứ từ dòng "Observation:" trở đi, vì mô hình có thể tự bịa Observation.
   - Nếu có cả Action và Final Answer, cái nào đứng trước thì thắng.
   - Phần JSON sau tên tool phải chịu được chữ thừa phía sau (mô hình hay viết
     thêm lời giải thích). Không có tham số thì coi là {}.
   - JSON hỏng, hoặc không phải object, thì trả kind="error" kèm thông điệp
     "Error: ..." chỉ rõ cách sửa, để đưa lại cho mô hình làm Observation.
   - Không có cả Action lẫn Final Answer cũng là "error".
2. Hàm `validate_action(specs, name, args) -> str | None`.
   - `specs` là danh sách ToolSpec (name, description, params với
     name/type/required) được dựng từ schema mà MCP server khai báo.
   - Trả None nếu hợp lệ. Nếu sai thì trả chuỗi "Error: ..." cho: tên tool không tồn tại (liệt kê các tool hợp lệ), thiếu tham số bắt buộc, thừa tham số lạ (liệt kê tham số được phép).
Ràng buộc: chỉ dùng thư viện chuẩn (re, json, dataclasses). Thông báo lỗi bằng tiếng Anh vì chúng sẽ trở thành Observation cho mô hình. Không gọi mạng. Không raise exception: mọi lỗi phải trả về dưới dạng giá trị.
```

**AI-generated corresponding code:**

```python
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
```

### 2. Print summary statistics of the testset (total, per type, per scope, RAGAS count) to the terminal

**Prompt:**

```text
Ở trong eval.py, sau khi load_testset đã kiểm tra hợp lệ, viết hàm summarize_testset(items) chỉ để in thống kê, không trả về gì. Cần in: tổng số câu; số câu theo type (answerable / no_answer); số câu có ragas = true; và với riêng câu no_answer thì số câu theo scope (in_domain / out_of_domain). items là list dict, mỗi dict có các khóa id, type, ragas,
và scope (chỉ có ở câu no_answer).
```

**AI-generated corresponding code:**

```python
def summarize_testset(items: list[dict]) -> None:
    by_type = Counter(it["type"] for it in items)
    by_scope = Counter(it["scope"] for it in items if it["type"] == "no_answer")
    n_ragas = sum(1 for it in items if it["ragas"])
    print(f"Testset: {len(items)} câu | answerable={by_type['answerable']} "
          f"no_answer={by_type['no_answer']} | chấm RAGAS={n_ragas}")
    for scope, n in sorted(by_scope.items()):
        print(f"  no_answer/{scope}: {n}")
```