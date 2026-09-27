"""C06 checks (EXPERIMENT_PLAN.md §9.3): the AgentDojo workspace adapter. All offline.

The headline check: every user task's ground truth, executed through our adapter
with GROUND_TRUTH_OUTPUT as the model output, passes AgentDojo's own utility
check, and every injection task's reference solution passes its security check.
Around it: the adapter is byte-for-byte AgentDojo's own execution, every value in
a tool result is labelled (every injection vector lands in an `untrusted` field),
the zone resolver, and determinism across processes and under replay.
"""
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.agent_pipeline.llms.openai_llm import _function_to_openai
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.types import get_text_content_as_str
from pydantic import BaseModel

import agentdojo_tools as adt
from agent import AgentSpec
from agentdojo_tools import (ANNOTATIONS, LABELS, SIDE_EFFECTS, ZONE_KINDS, attach, clocked,
                             control_values, internal_domain, new_workspace, rebuild,
                             resolve_zone, security, utility, workspace_registry,
                             workspace_suite)
from events import load, verify_records
from graph import AgentGraph
from injection_witnesses import WITNESSES
from run_context import RunContext
from test_c02_isolation import ScriptedClient
from test_c03_cache import StubOpenAI, make
from tools import RuntimeFault, ToolCtx

ROOT = Path(__file__).resolve().parents[1]
SUITE = workspace_suite()
USER_TASKS = sorted(SUITE.user_tasks, key=lambda t: int(t.rsplit("_", 1)[1]))
INJECTION_TASKS = sorted(SUITE.injection_tasks, key=lambda t: int(t.rsplit("_", 1)[1]))
CANARIES = {k: f"---CANARY_{k}---" for k in SUITE.get_injection_vector_defaults()}
ENVS = {"default": {}, "canary": CANARIES}


def jsonable(result):
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    if isinstance(result, list):
        return [jsonable(x) for x in result]
    return result


def run_calls(user_task_id, injections, calls_fn):
    """Executes calls_fn(env) through the registry, as agents do. Returns the run
    context, its workspace, and per call (call, text, observation records)."""
    ctx = RunContext(ScriptedClient(0), [], toolset="workspace")
    ws = attach(ctx, user_task_id, injections)
    out = []
    for k, call in enumerate(calls_fn(ws.env)):
        tctx = ToolCtx(run=ctx, agent_idx=0, agent_name="GT", round_idx=0, ref=f"{k}.0",
                       emit=ctx.events.emit)
        text = asyncio.run(ctx.tools.call(call.function, dict(call.args), tctx))
        out.append((call, text, [ctx.events.artifacts[i] for i in tctx.observations]))
    return ctx, ws, out


def errors(out):
    return [text for _, text, obs in out if any(o["field"] == "error" for o in obs)]


# ------------------------------------------------------------ the toolset
def test_toolset_is_agentdojos_24_tools_with_its_own_schemas():
    reg = workspace_registry()
    assert reg.names() == sorted(f.name for f in SUITE.tools) and len(reg.names()) == 24
    for f in SUITE.tools:
        assert json.loads(json.dumps(reg.specs([f.name])[0])) == \
            json.loads(json.dumps(_function_to_openai(f)))
        assert reg._tools[f.name].fn.__name__ == f.name
    assert SUITE.benchmark_version == (1, 2, 2)
    assert len(USER_TASKS) == 40 and len(INJECTION_TASKS) == 14


def test_annotation_table_covers_every_tool_exactly_once():
    assert set(ANNOTATIONS) == {f.name for f in SUITE.tools}
    params = {f.name: f.parameters.model_json_schema()["properties"] for f in SUITE.tools}
    for name, a in ANNOTATIONS.items():
        assert a.side_effect in SIDE_EFFECTS, name
        for c in a.control_args:
            assert c.arg in params[name], (name, c.arg)
            assert c.kind in ZONE_KINDS
        if a.side_effect in ("external_write", "irreversible"):
            assert a.control_args, f"{name} has an effect but no target"
    # the plan's named examples (EXPERIMENT_PLAN.md §2.2)
    assert ANNOTATIONS["send_email"].side_effect == "external_write"
    assert {c.arg for c in ANNOTATIONS["send_email"].control_args} >= {"recipients", "cc", "bcc"}
    assert ANNOTATIONS["share_file"].side_effect == "external_write"
    assert {c.arg for c in ANNOTATIONS["share_file"].control_args} == {"file_id", "email"}
    assert ANNOTATIONS["delete_file"].side_effect == "irreversible"
    assert ANNOTATIONS["cancel_calendar_event"].side_effect == "irreversible"
    assert "participants" in {c.arg for c in
                              ANNOTATIONS["add_calendar_event_participants"].control_args}


# --------------------------------------------------------------- zones
def test_internal_domain_comes_from_the_suites_account():
    env = new_workspace().env
    assert internal_domain() == "bluesparrowtech.com"
    for account in (env.inbox.account_email, env.calendar.account_email,
                    env.cloud_drive.account_email):
        assert account.split("@")[1] == internal_domain()


@pytest.mark.parametrize("value,zone", [
    ("emma.johnson@bluesparrowtech.com", "internal"),
    ("Emma.Johnson@BlueSparrowTech.COM", "internal"),
    ("  david.smith@bluesparrowtech.com ", "internal"),
    ("David Smith <david.smith@bluesparrowtech.com>", "internal"),
    ("mailto:david.smith@bluesparrowtech.com", "internal"),
    ("mark.black-2134@gmail.com", "external"),
    ("linda.green@luesparrowtech.com", "external"),            # lookalike, in the suite's data
    ("x@bluesparrowtech.com.evil.com", "external"),
    ("x@evilbluesparrowtech.com", "external"),
    ("x@mail.bluesparrowtech.com", "external"),                # subdomains are not the domain
    ("bluesparrowtech.com", "external"),
    ("x@y@bluesparrowtech.com", "external"),
    ("x @bluesparrowtech.com", "external"),
    ("@bluesparrowtech.com", "external"),
    ("", "external"),
    (None, "external"),
    (42, "external"),
])
def test_resolve_zone_for_addresses(value, zone):
    assert resolve_zone(value, "email") == zone


def test_ids_are_always_internal_and_unknown_kinds_raise():
    for kind in ("file_id", "event_id", "email_id"):
        assert resolve_zone("13", kind) == "internal"
        assert resolve_zone("anything@evil.com", kind) == "internal"
    with pytest.raises(ValueError):
        resolve_zone("x@y.com", "phone")


def test_every_address_in_the_environment_resolves_by_its_domain():
    env = new_workspace().env
    addrs = set()
    for e in env.inbox.emails.values():
        addrs.update([e.sender, *e.recipients, *e.cc, *e.bcc])
    for f in env.cloud_drive.files.values():
        addrs.update([f.owner, *f.shared_with])
    for ev in env.calendar.events.values():
        addrs.update(ev.participants)
    internal = {a for a in addrs if resolve_zone(a, "email") == "internal"}
    assert internal == {a for a in addrs if a.lower().endswith("@bluesparrowtech.com")}
    assert "emma.johnson@bluesparrowtech.com" in internal and len(addrs - internal) > 10


def test_control_values_with_zones():
    vals = control_values("send_email", {
        "recipients": "['a@bluesparrowtech.com', 'b@gmail.com']",   # parsed as AgentDojo does
        "subject": "s", "body": "b", "cc": ["c@bluesparrowtech.com"],
        "attachments": [{"type": "file", "file_id": "13"},
                        {"type": "event", "event_details": {"title": "t"}}]})
    assert [(v["arg"], v["index"], v["value"], v["zone"]) for v in vals] == [
        ("recipients", 0, "a@bluesparrowtech.com", "internal"),
        ("recipients", 1, "b@gmail.com", "external"),
        ("cc", 0, "c@bluesparrowtech.com", "internal"),
        ("attachments", 0, "13", "internal")]
    assert control_values("share_file", {"file_id": "3", "email": "x@evil.com",
                                         "permission": "rw"}) == [
        {"arg": "file_id", "index": None, "kind": "file_id", "value": "3", "zone": "internal"},
        {"arg": "email", "index": None, "kind": "email", "value": "x@evil.com", "zone": "external"}]
    assert control_values("search_emails", {"query": "x@evil.com"}) == []


# ------------------------------------------------- ground truth (the check)
@pytest.mark.parametrize("env_name", list(ENVS))
@pytest.mark.parametrize("task_id", USER_TASKS)
def test_user_task_ground_truth_passes_utility_through_the_adapter(task_id, env_name):
    task = SUITE.user_tasks[task_id]
    _, ws, out = run_calls(task_id, ENVS[env_name], task.ground_truth)
    assert not errors(out)
    assert utility(ws, task.GROUND_TRUTH_OUTPUT)


@pytest.mark.parametrize("env_name", list(ENVS))
@pytest.mark.parametrize("task_id", INJECTION_TASKS)
def test_injection_task_reference_solution_passes_security(task_id, env_name):
    """AgentDojo's own protocol (TaskSuite.check): the injection task's ground truth
    runs in the first user task's environment. Tasks 6-13 ship an empty ground
    truth, so they use our witnesses (injection_witnesses.py)."""
    task = SUITE.injection_tasks[task_id]
    first = USER_TASKS[0]
    if task_id in WITNESSES:
        assert task.ground_truth(new_workspace(first).env) == []
        calls_fn = WITNESSES[task_id]
    else:
        calls_fn = task.ground_truth
    _, ws, out = run_calls(first, ENVS[env_name], calls_fn)
    assert out and not errors(out)
    assert security(ws, task, task.GROUND_TRUTH_OUTPUT)


def test_doing_nothing_meets_no_goal_and_only_answer_only_tasks():
    """With no tool call: no injection goal is met, no user task passes with an empty
    answer, and the tasks that pass on the right answer alone (17 of 40) are the ones
    whose ground truth only reads. For those, utility is judged on the answer text."""
    _, ws, _ = run_calls(USER_TASKS[0], {}, lambda env: [])
    assert not any(security(ws, t, "") for t in INJECTION_TASKS)
    assert not any(utility(new_workspace(t), "") for t in USER_TASKS)
    answer_only = {t for t in USER_TASKS
                   if utility(new_workspace(t), SUITE.user_tasks[t].GROUND_TRUTH_OUTPUT)}
    read_only = {t for t in USER_TASKS
                 if all(ANNOTATIONS[c.function].side_effect == "read_only"
                        for c in SUITE.user_tasks[t].ground_truth(new_workspace(t).env))}
    assert answer_only <= read_only and len(answer_only) == 17


def test_every_user_task_starts_from_the_same_environment():
    """So the injection check above, done in one user task's environment as
    AgentDojo does, covers every pairing."""
    base = new_workspace().env.model_dump_json()
    assert all(new_workspace(t).env.model_dump_json() == base for t in USER_TASKS)


# --------------------------------------------- the adapter is AgentDojo
@pytest.mark.parametrize("task_id", USER_TASKS)
def test_adapter_reproduces_agentdojos_own_execution(task_id):
    """AgentDojo's GroundTruthPipeline, on the same clock, gives the same tool
    texts and the same final environment; the observations rebuild each raw result."""
    task = SUITE.user_tasks[task_id]
    _, ws, out = run_calls(task_id, CANARIES, task.ground_truth)

    native = new_workspace(task_id, CANARIES)
    with clocked(native):
        _, _, env, msgs, _ = GroundTruthPipeline(task).query(
            task.PROMPT, FunctionsRuntime(SUITE.tools), native.env)
    assert [text for _, text, _ in out] == \
        [get_text_content_as_str(m["content"]) for m in msgs if m["role"] == "tool"]
    assert ws.env.model_dump_json() == env.model_dump_json()

    raw = new_workspace(task_id, CANARIES)
    with clocked(raw):
        for (call, text, obs) in out:
            result, _ = adt._runtime().run_function(raw.env, call.function, call.args,
                                                    raise_on_error=True)
            assert text == tool_result_to_str(result)
            assert rebuild(obs) == jsonable(result)


# ---------------------------------------------------------- field labels
def test_injection_vectors_land_only_in_untrusted_fields_and_every_task_sees_one():
    seen = {}
    for task_id in USER_TASKS:
        _, _, out = run_calls(task_id, CANARIES, SUITE.user_tasks[task_id].ground_truth)
        hits = [o for _, _, obs in out for o in obs
                if isinstance(o["content"], str) and "---CANARY_" in o["content"]]
        assert all(o["label"] == "untrusted" for o in hits), task_id
        seen[task_id] = {(o["object"]["type"], o["field"]) for o in hits}
    assert all(seen.values()), [t for t, s in seen.items() if not s]     # all injectable
    assert set().union(*seen.values()) == {("Email", "body"), ("CalendarEvent", "description"),
                                           ("CloudDriveFile", "content")}


def _obs(out, k):
    return out[k][2]


def _find(obs, **match):
    return [o for o in obs if all(o.get(k) == v for k, v in match.items())]


def test_labels_attested_untrusted_agent_written_runtime():
    attacker = "mark.black-2134@gmail.com"
    from agentdojo.functions_runtime import FunctionCall as FC
    calls = [
        FC(function="send_email", args={"recipients": [attacker], "subject": "s", "body": "b"}),
        FC(function="get_sent_emails", args={}),
        FC(function="get_received_emails", args={}),
        FC(function="share_file", args={"file_id": "0", "email": attacker, "permission": "r"}),
        FC(function="create_calendar_event", args={"title": "t", "start_time": "2024-05-20 10:00",
                                                   "end_time": "2024-05-20 11:00",
                                                   "participants": [attacker]}),
        FC(function="get_current_day", args={}),
        FC(function="delete_email", args={"email_id": "0"}),
        FC(function="get_file_by_id", args={"file_id": "no-such-file"}),
        FC(function="search_contacts_by_name", args={"query": "david"}),
    ]
    _, ws, out = run_calls(USER_TASKS[0], {}, lambda env: calls)
    new_id = "34"
    # the email the team sent: the service stamps id, sender, time; the agent chose the rest
    sent = _obs(out, 0)
    assert {o["field"]: o["label"] for o in sent if o["path"][-1:] != [0]
            and o["field"] in ("id_", "sender", "timestamp", "status", "subject", "body")} == {
        "id_": "attested", "sender": "attested", "timestamp": "attested", "status": "attested",
        "subject": "agent_written", "body": "agent_written"}
    assert _find(sent, field="recipients")[0]["label"] == "agent_written"
    # read back later, the attacker's address is still the agent's, not the service's
    read_back = _obs(out, 1)
    assert _find(read_back, field="recipients", content=attacker)[0]["label"] == "agent_written"
    assert all(o["label"] == "attested" for o in _find(read_back, field="recipients")
               if o["object"]["id"] != new_id)
    assert all(o["label"] == "untrusted" for o in _find(read_back, field="body")
               if o["object"]["id"] != new_id)
    received = _obs(out, 2)
    assert {o["label"] for o in _find(received, field="sender")} == {"attested"}
    assert {o["label"] for o in _find(received, field="subject")} == {"untrusted"}
    shared = _find(_obs(out, 3), field="shared_with")
    assert [o["label"] for o in shared if o["key"] == attacker] == ["agent_written"]
    assert all(o["label"] == "attested" for o in shared if o["key"] != attacker)
    event = _obs(out, 4)
    assert {o["content"]: o["label"] for o in _find(event, field="participants")} == {
        attacker: "agent_written", "emma.johnson@bluesparrowtech.com": "agent_written"}
    assert _find(event, field="title")[0]["label"] == "agent_written"
    assert [(o["label"], o["content"]) for o in _obs(out, 5)] == [("attested", "2024-05-15")]
    assert [o["label"] for o in _obs(out, 6)] == ["runtime"]           # echoes its argument
    assert [(o["label"], o["field"]) for o in _obs(out, 7)] == [("runtime", "error")]
    assert {o["label"] for o in _obs(out, 8)} == {"attested"}
    for _, _, obs in out:
        assert {o["label"] for o in obs} <= set(LABELS)
        assert all(o["tool_result"].startswith("tr:") for o in obs)


def test_unlabelled_result_types_and_fields_fail_loudly():
    class Unknown(BaseModel):
        x: int = 1
    with pytest.raises(RuntimeFault):
        adt.observe("t", Unknown(), None, new_workspace().env)
    with pytest.raises(RuntimeFault):
        adt.observe("t", 3.5, None, new_workspace().env)


# -------------------------------------------------------- clock and runs
def test_clock_replaces_wall_time_inside_tool_calls_only():
    from agentdojo.functions_runtime import FunctionCall as FC
    calls = [FC(function="send_email", args={"recipients": ["a@b.com"], "subject": "s", "body": "b"}),
             FC(function="create_file", args={"filename": "f.txt", "content": "x"}),
             FC(function="append_to_file", args={"file_id": "0", "content": "more"})]
    _, ws, out = run_calls(USER_TASKS[0], {}, lambda env: calls)
    pre = ws.pre_environment
    latest = max([e.timestamp for e in pre.inbox.emails.values()] +
                 [f.last_modified for f in pre.cloud_drive.files.values()])
    assert ws.clock_start > latest
    stamps = [ws.env.inbox.emails["34"].timestamp, ws.env.cloud_drive.files["26"].last_modified,
              ws.env.cloud_drive.files["0"].last_modified]
    assert stamps == [ws.clock_start + adt._dt.timedelta(minutes=k) for k in range(3)]
    assert str(ws.clock_start.year) in out[0][1] and "2026" not in "".join(t for _, t, _ in out)
    # outside our calls AgentDojo's modules see the real clock
    from agentdojo.default_suites.v1.tools import email_client
    assert abs((email_client.datetime.datetime.now() - adt._dt.datetime.now()).total_seconds()) < 5


def test_workspaces_are_isolated_and_pre_environment_stays_frozen():
    a, b = new_workspace(USER_TASKS[0]), new_workspace(USER_TASKS[0])
    assert a.env is not b.env and a.env.inbox is not b.env.inbox
    before = a.pre_environment.model_dump_json()
    ctx = RunContext(ScriptedClient(0), [], toolset="workspace")
    ctx.workspace = a
    tctx = ToolCtx(run=ctx, agent_idx=0, agent_name="A", round_idx=0, ref="0.0", emit=ctx.events.emit)
    asyncio.run(ctx.tools.call("delete_file", {"file_id": "0"}, tctx))
    assert "0" not in a.env.cloud_drive.files and "0" in b.env.cloud_drive.files
    assert a.pre_environment.model_dump_json() == before


def test_outcome_checks_do_not_change_the_run_and_can_be_repeated():
    task = SUITE.injection_tasks["injection_task_6"]      # its check calls get_unread()
    _, ws, _ = run_calls(USER_TASKS[0], {}, WITNESSES["injection_task_6"])
    pre, post = ws.pre_environment.model_dump_json(), ws.env.model_dump_json()
    assert [security(ws, task, "") for _ in range(3)] == [True] * 3
    assert (ws.pre_environment.model_dump_json(), ws.env.model_dump_json()) == (pre, post)


def test_workspace_tools_need_a_workspace_and_a_workspace_run():
    ctx = RunContext(ScriptedClient(0), [], toolset="workspace")
    tctx = ToolCtx(run=ctx, agent_idx=0, agent_name="A", round_idx=0, ref="0.0", emit=ctx.events.emit)
    with pytest.raises(RuntimeFault):                   # a runtime bug, never shown to the model
        asyncio.run(ctx.tools.call("list_files", {}, tctx))
    with pytest.raises(RuntimeFault):
        attach(RunContext(ScriptedClient(0), []))      # builtin toolset
    attach(ctx)
    with pytest.raises(RuntimeFault):
        attach(ctx)


def _subprocess(code: str, **env) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                          text=True, env={**os.environ, **env}, timeout=300)


def test_adapter_refuses_randomized_hashing():
    p = _subprocess("import agentdojo_tools as a; a.new_workspace()", PYTHONHASHSEED="random")
    assert p.returncode != 0 and "PYTHONHASHSEED=0" in p.stderr


_DIGEST = """
import asyncio, hashlib, json, sys
sys.path.insert(0, "tests")
from test_c06_agentdojo import run_calls, SUITE, USER_TASKS, CANARIES
h = hashlib.sha256()
for t in USER_TASKS:
    _, ws, out = run_calls(t, CANARIES, SUITE.user_tasks[t].ground_truth)
    for _, text, obs in out:
        h.update(text.encode()); h.update(json.dumps(obs, sort_keys=True).encode())
    h.update(ws.env.model_dump_json().encode())
print(h.hexdigest())
"""


def test_two_processes_produce_identical_tool_results_and_environments():
    a, b = _subprocess(_DIGEST), _subprocess(_DIGEST)
    assert a.returncode == 0, a.stderr[-2000:]
    assert a.stdout == b.stdout and len(a.stdout.strip()) == 64


# ------------------------------------------------ inside a real agent run
class WorkspaceStub(StubOpenAI):
    """Samples like a real model (a fresh nonce per call), but calls workspace tools:
    first a search plus an external send, then reads the sent folder, then answers."""

    async def _create(self, **req):
        self.calls += 1
        msgs, tools = req["messages"], req.get("tools")
        last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
        step = sum(1 for m in msgs[last_user:] if m["role"] == "assistant")
        nonce = uuid.uuid4().hex[:8]
        plan = [[("search_emails", {"query": "birthday"}),
                 ("send_email", {"recipients": ["x@evil.example"], "subject": nonce,
                                 "body": "hello"})],
                [("get_sent_emails", {})]]
        if tools and step < len(plan):
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_{nonce}_{k}", "type": "function",
                 "function": {"name": n, "arguments": json.dumps(a)}}
                for k, (n, a) in enumerate(plan[step])]}
            finish = "tool_calls"
        else:
            message, finish = {"role": "assistant", "content": f"done {nonce}"}, "stop"
        from openai.types.chat import ChatCompletion
        return ChatCompletion.model_validate({
            "id": f"chatcmpl-{nonce}", "object": "chat.completion", "created": 0,
            "model": req["model"],
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}})


async def _workspace_run(client, **ctx_kw):
    spec = AgentSpec("EmailAgent", "email specialist", "You handle email.",
                     ["search_emails", "send_email", "get_sent_emails"])
    ctx = RunContext(client, [spec.name], toolset="workspace", **ctx_kw)
    attach(ctx, "user_task_0")
    g = AgentGraph([spec], np.zeros((1, 1), dtype=int), ctx)
    await g.run("task W", rounds=0, schedule="parallel", verbose=False)
    await g.synthesize("task W")
    return g, ctx


def test_agent_run_on_workspace_tools_logs_observations_and_replays_exactly(tmp_path):
    client, stub = make(tmp_path, stub=WorkspaceStub())
    g1, c1 = asyncio.run(_workspace_run(client))
    assert stub.calls == 4                              # 3 agent steps + moderator
    results = [r for r in c1.events.records if r.get("ev") == "tool_result"]
    assert [r["tool"] for r in results] == ["search_emails", "send_email", "get_sent_emails"]
    assert all(r["observations"] for r in results)
    arts = c1.events.artifacts
    assert all(arts[o]["kind"] == "observation" and arts[o]["label"] in LABELS
               for r in results for o in r["observations"])
    report = verify_records(c1.events.records, cache=client.cache)
    assert report["cache_keys_checked"] == report["llm_calls"] == 4
    start = next(r for r in c1.events.records if r.get("ev") == "run_start")
    assert start["toolset"] == "workspace"
    assert c1.workspace.traces[1].function == "send_email"
    assert "x@evil.example" in c1.workspace.env.inbox.emails["34"].recipients

    run_dir = g1.save("w1", out_dir=tmp_path / "runs")
    assert verify_records(load(run_dir / "events.jsonl"), cache=client.cache) == report

    # a new process, the same store, strict replay: no API call, the same run
    client2, stub2 = make(tmp_path, stub=WorkspaceStub())
    g2, c2 = asyncio.run(_workspace_run(client2, cache_mode="replay-strict"))
    assert stub2.calls == 0 and c2.cache_stats.chat_misses == 0
    assert json.dumps(g2.transcript()) == json.dumps(g1.transcript())
    # identical logs, except that the replay's calls were served from the cache
    blob = lambda c: "".join(json.dumps({k: v for k, v in r.items() if k != "cache_hit"})
                             for r in c.events.records)
    assert blob(c2) == blob(c1)
    assert {r["cache_hit"] for r in c2.events.records if r.get("ev") == "llm_call"} == {True}
    assert c2.workspace.env.model_dump_json() == c1.workspace.env.model_dump_json()


def test_an_agent_cannot_be_given_a_tool_outside_its_runs_toolset():
    ctx = RunContext(ScriptedClient(0), ["A"], toolset="workspace")
    with pytest.raises(ValueError, match="not in toolset"):
        AgentGraph([AgentSpec("A", "r", "i", ["blackboard_write"])], np.zeros((1, 1), dtype=int), ctx)
