"""Real, executing tools with dummy-but-consistent data.

Every tool here actually runs: the calculator evaluates, the SQL tool queries a
real sqlite database, the file tools touch a sandboxed directory, kb_search
retrieves from a real local corpus. `send_email` is the one deliberate
simulation -- it writes to an outbox file instead of sending, so side effects
are observable without being real.
"""
from __future__ import annotations

import ast
import asyncio
import json
import operator
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from config import CFG


# ============================================================= registry types
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON schema
    fn: Callable[..., Any]    # sync or async

    def spec(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": self.parameters}}


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, name, description, parameters):
        def deco(fn):
            self._tools[name] = Tool(name, description, parameters, fn)
            return fn
        return deco

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, names: list[str] | None = None) -> list[dict]:
        names = names if names is not None else self.names()
        return [self._tools[n].spec() for n in names if n in self._tools]

    async def call(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps({"error": f"unknown tool '{name}'"})
        try:
            result = tool.fn(**args)
            if asyncio.iscoroutine(result):
                result = await result
        except TypeError as e:
            return json.dumps({"error": f"bad arguments for {name}: {e}"})
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})
        return result if isinstance(result, str) else json.dumps(result, default=str)


REGISTRY = ToolRegistry()


# ================================================================ calculator
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("only arithmetic over numeric literals is allowed")


@REGISTRY.register(
    "calculator", "Evaluate an arithmetic expression, e.g. '(1200 * 0.15) + 340'.",
    {"type": "object",
     "properties": {"expression": {"type": "string", "description": "Arithmetic expression."}},
     "required": ["expression"]})
def calculator(expression: str):
    value = _safe_eval(ast.parse(expression, mode="eval"))
    return {"expression": expression, "value": value}


# ======================================================================= sql
_DB: sqlite3.Connection | None = None

_SEED = """
CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, team TEXT, role TEXT, salary INTEGER, hired TEXT);
CREATE TABLE products  (sku TEXT PRIMARY KEY, name TEXT, category TEXT, unit_price REAL);
CREATE TABLE sales     (id INTEGER PRIMARY KEY, sku TEXT, region TEXT, units INTEGER, quarter TEXT);
INSERT INTO employees VALUES
 (1,'Dana Okafor','Robotics','Principal Engineer',186000,'2019-03-11'),
 (2,'Wei Zhang','Robotics','Senior Engineer',154000,'2021-07-01'),
 (3,'Priya Raman','Perception','Research Lead',192000,'2018-01-22'),
 (4,'Tom Bergstrom','Perception','Engineer',131000,'2022-09-15'),
 (5,'Aisha Bello','Platform','Staff Engineer',171000,'2020-05-04'),
 (6,'Luis Moreno','Platform','Engineer',128000,'2023-02-20'),
 (7,'Hannah Fox','Sales','Account Director',145000,'2020-11-09'),
 (8,'Kenji Sato','Sales','Account Manager',112000,'2023-06-01');
INSERT INTO products VALUES
 ('NR-100','Arm 100','manipulator',24500.0),
 ('NR-220','Arm 220 HD','manipulator',41200.0),
 ('NR-310','Vision Pod','perception',8800.0),
 ('NR-450','Fleet Controller','platform',15600.0);
INSERT INTO sales VALUES
 (1,'NR-100','NA',34,'2025Q1'),(2,'NR-100','EU',21,'2025Q1'),
 (3,'NR-220','NA',12,'2025Q1'),(4,'NR-310','NA',88,'2025Q1'),
 (5,'NR-310','APAC',64,'2025Q1'),(6,'NR-450','EU',19,'2025Q1'),
 (7,'NR-100','NA',41,'2025Q2'),(8,'NR-220','APAC',17,'2025Q2'),
 (9,'NR-310','EU',73,'2025Q2'),(10,'NR-450','NA',28,'2025Q2');
"""


def _db() -> sqlite3.Connection:
    global _DB
    if _DB is None:
        _DB = sqlite3.connect(":memory:", check_same_thread=False)
        _DB.executescript(_SEED)
        _DB.row_factory = sqlite3.Row
    return _DB


@REGISTRY.register(
    "sql_query",
    "Run a read-only SQL SELECT against the company database. "
    "Tables: employees(id,name,team,role,salary,hired), "
    "products(sku,name,category,unit_price), sales(id,sku,region,units,quarter).",
    {"type": "object",
     "properties": {"sql": {"type": "string", "description": "A single SELECT statement."}},
     "required": ["sql"]})
def sql_query(sql: str):
    stripped = sql.strip().rstrip(";")
    if not stripped.lower().startswith(("select", "with")):
        return {"error": "only SELECT / WITH queries are permitted"}
    if ";" in stripped:
        return {"error": "only a single statement is permitted"}
    rows = _db().execute(stripped).fetchmany(50)
    return {"sql": stripped, "row_count": len(rows), "rows": [dict(r) for r in rows]}


# ================================================================= kb_search
_KB_CACHE: list[tuple[str, str]] | None = None


def _load_kb() -> list[tuple[str, str]]:
    """Returns [(source, paragraph)] chunks from the knowledge_base directory."""
    global _KB_CACHE
    if _KB_CACHE is None:
        chunks = []
        for path in sorted(CFG.kb_dir.glob("*.md")):
            for para in path.read_text().split("\n\n"):
                para = para.strip()
                if len(para) > 40:
                    chunks.append((path.name, para))
        _KB_CACHE = chunks
    return _KB_CACHE


@REGISTRY.register(
    "kb_search", "Search the internal company knowledge base and return the best matching passages.",
    {"type": "object",
     "properties": {"query": {"type": "string"},
                    "k": {"type": "integer", "description": "How many passages (default 3)."}},
     "required": ["query"]})
def kb_search(query: str, k: int = 3):
    terms = {t for t in query.lower().split() if len(t) > 2}
    scored = []
    for source, para in _load_kb():
        words = para.lower()
        score = sum(words.count(t) for t in terms)
        if score:
            scored.append((score, source, para))
    scored.sort(key=lambda x: -x[0])
    hits = [{"source": s, "score": sc, "text": p} for sc, s, p in scored[:max(1, k)]]
    return {"query": query, "hits": hits or [{"note": "no matching passages"}]}


# =============================================================== file sandbox
def _resolve(rel: str) -> Path:
    CFG.workspace.mkdir(parents=True, exist_ok=True)
    target = (CFG.workspace / rel).resolve()
    if not str(target).startswith(str(CFG.workspace.resolve())):
        raise ValueError("path escapes the workspace sandbox")
    return target


@REGISTRY.register(
    "write_file", "Write text to a file in the shared workspace (overwrites).",
    {"type": "object",
     "properties": {"path": {"type": "string", "description": "Relative path, e.g. 'report.md'."},
                    "content": {"type": "string"}},
     "required": ["path", "content"]})
def write_file(path: str, content: str):
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return {"path": path, "bytes_written": len(content.encode())}


@REGISTRY.register(
    "read_file", "Read a text file from the shared workspace.",
    {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]})
def read_file(path: str):
    target = _resolve(path)
    if not target.exists():
        return {"error": f"no such file: {path}"}
    return {"path": path, "content": target.read_text()[:8000]}


@REGISTRY.register(
    "list_files", "List files currently in the shared workspace.",
    {"type": "object", "properties": {}, "required": []})
def list_files():
    CFG.workspace.mkdir(parents=True, exist_ok=True)
    files = [str(p.relative_to(CFG.workspace))
             for p in sorted(CFG.workspace.rglob("*")) if p.is_file()]
    return {"files": files}


# ============================================================ simulated email
@REGISTRY.register(
    "send_email", "Send an email. SIMULATED: appends to workspace/outbox.jsonl, nothing leaves the machine.",
    {"type": "object",
     "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
     "required": ["to", "subject", "body"]})
def send_email(to: str, subject: str, body: str):
    CFG.workspace.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.time(), "to": to, "subject": subject, "body": body}
    with (CFG.workspace / "outbox.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    return {"status": "queued (simulated)", "to": to, "subject": subject}
