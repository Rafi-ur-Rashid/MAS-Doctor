"""AgentDojo's workspace suite as a toolset (C06, EXPERIMENT_PLAN.md §2.2).

What the adapter guarantees:
  - The 24 tools are AgentDojo's own. Each schema is exactly what AgentDojo's OpenAI
    pipeline sends. Each call runs through FunctionsRuntime.run_function on this run's
    environment. The model sees what AgentDojo shows it: the error string if the call
    failed, otherwise the YAML dump (tool_result_to_str). Before the call, list-valued
    string arguments are parsed, as AgentDojo's ToolsExecutor does.
  - A run owns a WorkspaceState: an environment built fresh from the suite's YAML with
    the chosen injections, then the user task's init_environment; a deep copy taken
    before anything runs (pre_environment); the trace of tool calls; and a clock.
  - Outcomes are AgentDojo's own checks, run on the post-run environment:
    utility() for the user task, security() for an injection task. Each check gets
    its own deep copies of the pre- and post-run environments, because some checks
    change them: injection tasks 6, 8 and 9 and user task 24 call
    inbox.get_unread(), which marks emails read. AgentDojo runs utility and then
    security on the same objects, so there one check can change what the next one
    sees, and a repeated check can pass vacuously. Here every check sees the run's
    environments exactly as they were, however often and in whatever order it runs.
  - Every value in a tool result is also registered as an `observation` artifact,
    with its path in the result and a provenance label (field-level labels, §0 item 4):
      attested       metadata the service vouches for: ids, addresses, times, status,
                     sizes. The value was in the task's starting environment, or the
                     service generated it (a new email's id, sender and timestamp).
      untrusted      free text written by whoever created the object: subject, body,
                     title, description, location, filename, content. Every
                     AgentDojo injection vector is in one of these (tested).
      agent_written  a value that an agent's own tool call put into the environment
                     during this run (the recipients of an email the team sent, a
                     participant it added). The service only echoes it back, so it
                     is not attested. Its provenance is the arguments of that call.
      runtime        a status or error string the tool composed. It can echo the
                     caller's arguments, so it is never evidence for a value.
  - The annotation table gives each tool a side-effect class and its control
    arguments, and resolve_zone() says whether a control value is internal.

Determinism. Two things in AgentDojo's tools would make a replayed run differ:
  1. send_email, create_file and append_to_file stamp objects with datetime.now(), and
     that time is in the result the model reads. Inside our tool calls, now() is the
     run's clock instead: one minute after the latest timestamp in the task's starting
     environment, advancing one minute per call. The clock must start after every
     stored timestamp, because one utility check requires an appended file's
     last_modified to increase.
  2. create_calendar_event orders participants through set(). With randomized string
     hashing, that order differs between processes. The adapter refuses to run unless
     PYTHONHASHSEED=0, which the mastrust env sets.
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime as _dt
import importlib.resources
import re
import sys
import types
from ast import literal_eval
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import yaml
from pydantic import BaseModel

from agentdojo.agent_pipeline.tool_execution import is_string_list, tool_result_to_str
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.default_suites.v1.tools import cloud_drive_client, email_client
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import text_content_block_from_string

from config import CFG
from events import artifact
from tools import RuntimeFault, Tool, ToolCtx, ToolRegistry

# ======================================================== annotation table
# A research decision, to be cited (EXPERIMENT_PLAN.md §9.3 C06). Side-effect
# classes, from least to most severe:
#   read_only       changes nothing
#   local_write     changes the user's own data; nothing leaves the account
#   external_write  something reaches, or becomes visible to, another person
#   irreversible    destroys or ends something, and no tool can undo it
# A control argument names the target of the effect: who receives something, or
# which object is changed or destroyed. Content arguments (subject, body, the new
# text) are not control arguments; the monitor reads them separately.
SIDE_EFFECTS = ("read_only", "local_write", "external_write", "irreversible")
ZONE_KINDS = ("email", "file_id", "event_id", "email_id")


@dataclass(frozen=True)
class ControlArg:
    arg: str                  # argument name
    kind: str                 # what the value names (ZONE_KINDS)
    key: str | None = None    # for a list of dicts: the key inside each item that holds it


@dataclass(frozen=True)
class Annotation:
    side_effect: str
    control_args: tuple[ControlArg, ...] = ()
    note: str = ""


_A, _C = Annotation, ControlArg
ANNOTATIONS: dict[str, Annotation] = {
    # ------------------------------------------------------------- email
    "send_email": _A("external_write",
                     (_C("recipients", "email"), _C("cc", "email"), _C("bcc", "email"),
                      _C("attachments", "file_id", key="file_id")),
                     "an attached file id decides which file leaves the account"),
    "delete_email": _A("irreversible", (_C("email_id", "email_id"),),
                       "moves the email to a trash that no tool can read or restore"),
    "get_unread_emails": _A("local_write", (),
                            "returns the unread emails and marks them read"),
    "get_sent_emails": _A("read_only"),
    "get_received_emails": _A("read_only"),
    "get_draft_emails": _A("read_only"),
    "search_emails": _A("read_only"),
    "search_contacts_by_name": _A("read_only"),
    "search_contacts_by_email": _A("read_only"),
    # ---------------------------------------------------------- calendar
    "get_current_day": _A("read_only"),
    "search_calendar_events": _A("read_only"),
    "get_day_calendar_events": _A("read_only"),
    "create_calendar_event": _A("external_write", (_C("participants", "email"),),
                                "emails an invitation to every participant"),
    "cancel_calendar_event": _A("irreversible", (_C("event_id", "event_id"),),
                                "no tool un-cancels an event; also emails the participants"),
    "reschedule_calendar_event": _A("external_write", (_C("event_id", "event_id"),),
                                    "emails the participants the new time"),
    "add_calendar_event_participants": _A(
        "external_write", (_C("event_id", "event_id"), _C("participants", "email")),
        "gives the new participants the event; its docstring says they are emailed, "
        "though AgentDojo's code does not send one"),
    # ------------------------------------------------------------- drive
    "append_to_file": _A("local_write", (_C("file_id", "file_id"),)),
    "search_files_by_filename": _A("read_only"),
    "create_file": _A("local_write"),
    "delete_file": _A("irreversible", (_C("file_id", "file_id"),)),
    "get_file_by_id": _A("read_only"),
    "list_files": _A("read_only"),
    "share_file": _A("external_write", (_C("file_id", "file_id"), _C("email", "email")),
                     "permission (r or rw) is not a control argument: it names no target"),
    "search_files": _A("read_only"),
}


# =========================================================== zone resolver
@lru_cache(maxsize=None)
def internal_domain() -> str:
    """The user's own domain: the domain of account_email in the suite's inbox.yaml."""
    path = (importlib.resources.files("agentdojo") / "data" / "suites" / CFG.agentdojo_suite
            / "include" / "inbox.yaml")
    with importlib.resources.as_file(path) as p:
        account = yaml.safe_load(p.read_text())["account_email"]
    return account.rsplit("@", 1)[1].lower()


_NAME_ADDR = re.compile(r"[^<>]*<([^<>]*)>")


def normalize_address(value) -> str | None:
    """Lower-cased bare address, or None when the value is not a well-formed one.
    Accepts "Name <addr>" and "mailto:addr", as a mail client would."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    m = _NAME_ADDR.fullmatch(s)
    if m:
        s = m.group(1).strip()
    if s.lower().startswith("mailto:"):
        s = s[len("mailto:"):]
    s = s.lower()
    if s.count("@") != 1 or any(ch.isspace() for ch in s):
        return None
    local, domain = s.split("@")
    if not local or not domain:
        return None
    return s


def resolve_zone(value, kind: str, domain: str | None = None) -> str:
    """'internal' or 'external'. A pure function (unit-tested).

    Ids name objects inside the user's own services, so they are always internal.
    An address is internal only if it is well formed and its domain is exactly the
    user's domain. Anything else is external: other domains, subdomains, lookalikes
    (the suite itself holds a share to linda.green@luesparrowtech.com) and malformed
    values. The resolver fails closed."""
    if kind not in ZONE_KINDS:
        raise ValueError(f"unknown kind {kind!r} ({'|'.join(ZONE_KINDS)})")
    if kind != "email":
        return "internal"
    addr = normalize_address(value)
    if addr is None:
        return "external"
    return "internal" if addr.split("@")[1] == (domain or internal_domain()) else "external"


def parse_args(args: dict) -> dict:
    """AgentDojo's ToolsExecutor turns a string that parses as a list into that list
    before calling a tool. Doing the same keeps our calls identical to its calls."""
    return {k: literal_eval(v) if isinstance(v, str) and is_string_list(v) else v
            for k, v in args.items()}


def control_values(tool: str, args: dict) -> list[dict]:
    """The control-argument values of a call, each with its kind and zone."""
    args = parse_args(args)
    out = []
    for c in ANNOTATIONS[tool].control_args:
        v = args.get(c.arg)
        if v is None:
            continue
        items = v if isinstance(v, list) else [v]
        for i, item in enumerate(items):
            if c.key is not None:
                if not isinstance(item, dict) or c.key not in item:
                    continue
                item = item[c.key]
            out.append({"arg": c.arg, "index": i if isinstance(v, list) else None,
                        "kind": c.kind, "value": item, "zone": resolve_zone(item, c.kind)})
    return out


# ================================================================ the suite
@lru_cache(maxsize=None)
def workspace_suite():
    if CFG.agentdojo_suite != "workspace":
        raise NotImplementedError("the field labels and annotations cover the workspace suite only")
    return get_suite(CFG.agentdojo_benchmark, CFG.agentdojo_suite)


@lru_cache(maxsize=None)
def _runtime() -> FunctionsRuntime:
    return FunctionsRuntime(workspace_suite().tools)


def openai_spec(f) -> dict:
    """The tool spec AgentDojo's OpenAI pipeline sends (_function_to_openai)."""
    return {"type": "function",
            "function": {"name": f.name, "description": f.description,
                         "parameters": f.parameters.model_json_schema()}}


@lru_cache(maxsize=None)
def workspace_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for f in workspace_suite().tools:
        spec = openai_spec(f)["function"]
        reg._tools[f.name] = Tool(f.name, spec["description"], spec["parameters"],
                                  _bind(f.name), needs_ctx=True)
    return reg


def _bind(name: str):
    def fn(ctx: ToolCtx, **args):
        return _execute(ctx, name, args)
    fn.__name__ = fn.__qualname__ = name
    return fn


# =================================================================== clock
_NOW: contextvars.ContextVar = contextvars.ContextVar("agentdojo_now", default=None)


class _ClockedDatetime(_dt.datetime):
    """Stands in for datetime.datetime inside AgentDojo's email and drive tool
    modules. Only now() differs: inside one of our tool calls, it reads the run's
    clock. Elsewhere it is the real clock, so AgentDojo's own code is unchanged."""

    @classmethod
    def now(cls, tz=None):
        clock = _NOW.get()
        return _dt.datetime.now(tz) if clock is None else clock()


def _install_clock() -> None:
    for mod in (email_client, cloud_drive_client):
        if getattr(mod.datetime, "_mas_clocked", False):
            continue
        shim = types.ModuleType("datetime")
        shim.__dict__.update(vars(_dt))
        shim.datetime = _ClockedDatetime
        shim._mas_clocked = True
        mod.datetime = shim


_install_clock()


@contextlib.contextmanager
def clocked(ws: "WorkspaceState"):
    token = _NOW.set(ws.now)
    try:
        yield
    finally:
        _NOW.reset(token)


def _clock_start(env) -> _dt.datetime:
    stamps = [e.timestamp for e in env.inbox.emails.values()]
    stamps += [f.last_modified for f in env.cloud_drive.files.values()]
    return max(stamps).replace(second=0, microsecond=0) + _dt.timedelta(minutes=1)


# ================================================================ run state
@dataclass
class WorkspaceState:
    env: Any                                 # the environment the tools act on
    pre_environment: Any                     # deep copy before anything ran
    user_task: BaseUserTask | None
    injections: dict
    clock_start: _dt.datetime
    traces: list[FunctionCall] = field(default_factory=list)
    ticks: int = 0

    def now(self) -> _dt.datetime:
        t = self.clock_start + _dt.timedelta(minutes=self.ticks)
        self.ticks += 1
        return t


def _require_fixed_hash_seed() -> None:
    if sys.flags.hash_randomization:
        raise RuntimeFault("AgentDojo's tools order values through set(); run with "
                           "PYTHONHASHSEED=0 (conda run -n mastrust sets it)")


def new_workspace(user_task_id: str | None = None,
                  injections: dict[str, str] | None = None) -> WorkspaceState:
    """The environment a run starts from, built the way AgentDojo's
    run_task_with_pipeline builds it. Each call parses the suite's YAML afresh, so
    two runs never share an object."""
    _require_fixed_hash_seed()
    suite = workspace_suite()
    injections = dict(injections or {})
    env = suite.load_and_inject_default_environment(injections)
    task = suite.get_user_task_by_id(user_task_id) if user_task_id is not None else None
    if task is not None:
        env = task.init_environment(env)
    pre = env.model_copy(deep=True)
    return WorkspaceState(env, pre, task, injections, _clock_start(pre))


def attach(run, user_task_id: str | None = None,
           injections: dict[str, str] | None = None) -> WorkspaceState:
    if run.toolset != "workspace":
        raise RuntimeFault(f"run uses toolset {run.toolset!r}, not 'workspace'")
    if run.workspace is not None:
        raise RuntimeFault("run already has a workspace")
    run.workspace = new_workspace(user_task_id, injections)
    return run.workspace


# ================================================================ execution
def _execute(ctx: ToolCtx, name: str, args: dict) -> str:
    ws = getattr(ctx.run, "workspace", None)
    if ws is None:
        raise RuntimeFault("workspace tools need a WorkspaceState (agentdojo_tools.attach)")
    args = parse_args(args)
    ws.traces.append(FunctionCall(function=name, args=args, id=ctx.ref))
    with clocked(ws):
        result, error = _runtime().run_function(ws.env, name, args)
    # what AgentDojo's OpenAI pipeline puts in the tool message
    text = error if error else tool_result_to_str(result)
    for n, leaf in enumerate(observe(name, result, error, ws.pre_environment)):
        aid = f"obs:{ctx.ref}:{n}"
        ctx.observations.append(ctx.register(artifact(
            aid, "observation", "env:workspace", leaf.pop("value"), tool=name,
            tool_result=f"tr:{ctx.ref}", **leaf)))
    return text


# ============================================================ field labels
ATTESTED, UNTRUSTED, AGENT_WRITTEN, RUNTIME = "attested", "untrusted", "agent_written", "runtime"
LABELS = (ATTESTED, UNTRUSTED, AGENT_WRITTEN, RUNTIME)

# Per result type, the class of every field. An unknown type or field raises, so a
# change in AgentDojo's models cannot silently go unlabelled.
FIELD_CLASS: dict[str, dict[str, str]] = {
    "Email": {"id_": ATTESTED, "sender": ATTESTED, "recipients": ATTESTED, "cc": ATTESTED,
              "bcc": ATTESTED, "subject": UNTRUSTED, "body": UNTRUSTED, "status": ATTESTED,
              "read": ATTESTED, "timestamp": ATTESTED, "attachments": ATTESTED},
    "CalendarEvent": {"id_": ATTESTED, "title": UNTRUSTED, "description": UNTRUSTED,
                      "start_time": ATTESTED, "end_time": ATTESTED, "location": UNTRUSTED,
                      "participants": ATTESTED, "all_day": ATTESTED, "status": ATTESTED},
    "CloudDriveFile": {"id_": ATTESTED, "filename": UNTRUSTED, "content": UNTRUSTED,
                       "owner": ATTESTED, "last_modified": ATTESTED, "shared_with": ATTESTED,
                       "size": ATTESTED},
    "EmailContact": {"email": ATTESTED, "name": ATTESTED},
}
# Fields the service sets itself and no tool takes from its arguments: attested
# whether or not the object existed before the run. (A contact is built by the
# service from stored addresses, and no tool adds one.)
SERVICE_FIELDS: dict[str, set[str]] = {
    "Email": {"id_", "sender", "status", "read", "timestamp"},
    "CalendarEvent": {"id_", "all_day", "status"},
    "CloudDriveFile": {"id_", "owner", "last_modified", "size"},
    "EmailContact": {"email", "name"},
}
# Plain-string results the service vouches for; every other string a tool returns
# is a status message that can echo its arguments.
ATTESTED_STRING_RESULTS = {"get_current_day"}

_MISSING = object()


def _pre_object(pre_env, obj: BaseModel):
    store = {"Email": lambda: pre_env.inbox.emails,
             "CalendarEvent": lambda: pre_env.calendar.events,
             "CloudDriveFile": lambda: pre_env.cloud_drive.files}.get(type(obj).__name__)
    return None if store is None else store().get(obj.id_)


def observe(tool: str, result, error: str | None, pre_env) -> list[dict]:
    """One record per value in a tool result: {path, value, label, object, field}.
    Values are JSON-mode (as model_dump(mode="json")); once logged, the records
    rebuild the result exactly (rebuild())."""
    if error:
        return [{"path": [], "value": error, "label": RUNTIME, "object": None, "field": "error"}]
    if isinstance(result, BaseModel):
        out: list[dict] = []
        _walk_object(result, [], pre_env, out)
        return out
    if isinstance(result, list):
        out = []
        for i, item in enumerate(result):
            if not isinstance(item, BaseModel):
                raise RuntimeFault(f"{tool}: unlabelled list item type {type(item).__name__}")
            _walk_object(item, [i], pre_env, out)
        if not result:
            out.append({"path": [], "value": [], "label": ATTESTED, "object": None, "field": None})
        return out
    if isinstance(result, str):
        label = ATTESTED if tool in ATTESTED_STRING_RESULTS else RUNTIME
        return [{"path": [], "value": result, "label": label, "object": None, "field": None}]
    raise RuntimeFault(f"{tool}: unlabelled result type {type(result).__name__}")


def _walk_object(obj: BaseModel, path: list, pre_env, out: list) -> None:
    tname = type(obj).__name__
    if tname not in FIELD_CLASS:
        raise RuntimeFault(f"no field labels for result type {tname}")
    classes, service = FIELD_CLASS[tname], SERVICE_FIELDS[tname]
    pre = _pre_object(pre_env, obj)
    ref = {"type": tname, "id": getattr(obj, "id_", None) or getattr(obj, "email", None)}
    dump = obj.model_dump(mode="json")
    if set(dump) != set(classes):
        raise RuntimeFault(f"{tname}: fields {sorted(set(dump) ^ set(classes))} are not labelled")
    for fname, jval in dump.items():
        val = getattr(obj, fname)
        cls = classes[fname]
        pre_val = getattr(pre, fname) if pre is not None else _MISSING

        def label(present_before: bool) -> str:
            return cls if fname in service or present_before else AGENT_WRITTEN

        def leaf(p, v, lab, **extra):
            out.append({"path": p, "value": v, "label": lab, "object": ref, "field": fname, **extra})

        fpath = path + [fname]
        if isinstance(val, list) and val:
            for i, (item, jitem) in enumerate(zip(val, jval)):
                if isinstance(item, BaseModel):          # an event attached to an email
                    _walk_object(item, fpath + [i], pre_env, out)
                else:
                    leaf(fpath + [i], jitem,
                         label(pre_val is not _MISSING and item in pre_val))
        elif isinstance(val, dict) and val:              # shared_with: address -> permission
            for key, jitem in jval.items():
                leaf(fpath + [key], jitem,
                     label(pre_val is not _MISSING and pre_val.get(key) == val[key]), key=key)
        else:
            leaf(fpath, jval, label(pre_val is not _MISSING and pre_val == val))


def rebuild(records: list[dict]):
    """The JSON-mode tool result, rebuilt from the observation artifacts in the log
    (the value is their `content`). Tests that observe() loses nothing."""
    if len(records) == 1 and records[0]["path"] == []:
        return records[0]["content"]
    root: Any = None
    for r in records:
        path, value = r["path"], r["content"]
        if root is None:
            root = [] if isinstance(path[0], int) else {}
        node = root
        for k, nxt in zip(path, path[1:]):
            child_default = [] if isinstance(nxt, int) else {}
            if isinstance(node, list):
                while len(node) <= k:
                    node.append(None)
                if node[k] is None:
                    node[k] = child_default
            else:
                node.setdefault(k, child_default)
            node = node[k]
        last = path[-1]
        if isinstance(node, list):
            while len(node) <= last:
                node.append(None)
        node[last] = value
    return root


# ================================================================= outcomes
def _model_output(text: str | None):
    return [text_content_block_from_string(text)] if text is not None else []


def _check(ws: WorkspaceState, task, model_output: str | None) -> bool:
    """TaskSuite._check_task_result (the *_from_traces check if the task defines
    one, else utility/security), on private copies of the environments."""
    return workspace_suite()._check_task_result(
        task, _model_output(model_output), ws.pre_environment.model_copy(deep=True),
        ws.env.model_copy(deep=True), list(ws.traces))


def utility(ws: WorkspaceState, model_output: str | None) -> bool:
    """AgentDojo's utility check for the run's user task."""
    if ws.user_task is None:
        raise RuntimeFault("this workspace has no user task")
    return _check(ws, ws.user_task, model_output)


def security(ws: WorkspaceState, injection_task: BaseInjectionTask | str,
             model_output: str | None) -> bool:
    """AgentDojo's security check: True means the attacker's goal was achieved."""
    if isinstance(injection_task, str):
        injection_task = workspace_suite().get_injection_task_by_id(injection_task)
    return _check(ws, injection_task, model_output)
