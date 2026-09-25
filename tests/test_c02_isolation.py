"""C02 checks (EXPERIMENT_PLAN.md §9.3): per-run isolation and repeatability.

All offline. A scripted model stands in for the LLM: it writes to the shared
blackboard, writes files, sends email, reads the blackboard back, and sleeps a
random time before every reply, so any dependence on API latency or on another
run's state shows up as a changed transcript.
"""
import asyncio
import hashlib
import json
import random
import re
import time

import numpy as np
import pytest

import topology
from agent import AgentSpec
from config import CFG, Config
from graph import AgentGraph
from llm import LLMClient, Usage, _hash_embed
from run_context import EMPTY, RunContext, SnapshotError
from team import build_team
from tools import REGISTRY, ToolCtx


# ------------------------------------------------------------ scripted model
class _Fn:
    def __init__(self, name, args):
        self.name, self.arguments = name, json.dumps(args)


class _Call:
    def __init__(self, cid, name, args):
        self.id, self.type, self.function = cid, "function", _Fn(name, args)


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.role, self.content, self.tool_calls = "assistant", content, tool_calls


class ScriptedClient:
    """Deterministic in content, random in latency."""
    embed_space = "hash-256"

    def __init__(self, seed: int, max_delay: float = 0.01, always_tools: bool = False):
        self.rng = random.Random(seed)
        self.max_delay = max_delay
        self.always_tools = always_tools
        self.prompts: list[list[dict]] = []

    def for_run(self, usage: Usage, **cache_settings):   # cache settings unused offline
        return _ScriptedRun(self, usage)


class _ScriptedRun:
    def __init__(self, client, usage):
        self.client, self.usage = client, usage

    @property
    def embed_space(self):
        return self.client.embed_space

    async def embed(self, texts):
        await asyncio.sleep(self.client.rng.random() * self.client.max_delay)
        return [_hash_embed(t) for t in texts]

    async def chat(self, messages, tools=None):
        c = self.client
        c.prompts.append(json.loads(json.dumps(messages)))
        await asyncio.sleep(c.rng.random() * c.max_delay)
        m = re.match(r"You are agent_(\d+) \(([^)]+)\)", messages[0]["content"])
        if m is None or not tools:                      # moderator, or forced answer
            return _Msg(content=f"final after {len(messages)} messages")
        name = m.group(2)
        last_user = max(i for i, x in enumerate(messages) if x["role"] == "user")
        step = sum(1 for x in messages[last_user:] if x["role"] == "assistant")
        tag = hashlib.sha256(messages[last_user]["content"].encode()).hexdigest()[:8]
        names = {t["function"]["name"] for t in tools}
        cid = lambda k: f"call_{name}_{len(messages)}_{k}"
        if c.always_tools:
            return _Msg(tool_calls=[_Call(cid(0), "blackboard_read", {})])
        if step == 0:
            calls = [_Call(cid(0), "blackboard_write", {"key": f"{name}_{tag}", "value": f"{name}:{tag}"})]
            if "write_file" in names:
                calls.append(_Call(cid(1), "write_file", {"path": "shared.txt", "content": f"{name}:{tag}"}))
            if "send_email" in names:
                calls.append(_Call(cid(2), "send_email", {"to": "x@y.z", "subject": tag, "body": name}))
            return _Msg(tool_calls=calls)
        if step == 1:
            return _Msg(tool_calls=[_Call(cid(0), "blackboard_read", {})])
        seen = messages[-1]["content"]
        return _Msg(content=f"{name} saw {seen}")


# ------------------------------------------------------------------ helpers
async def _one_run(client, task, schedule="parallel", state_from=EMPTY, state_dir=None,
                   topo="complete", rounds=2):
    specs = build_team(5)
    ctx = RunContext(client, [s.name for s in specs], state_from=state_from, state_dir=state_dir)
    g = AgentGraph(specs, topology.build_adjacency(topo, len(specs)), ctx)
    await g.run(task, rounds=rounds, schedule=schedule, verbose=False)
    await g.synthesize(task)
    return g, ctx


def _bytes(g: AgentGraph) -> bytes:
    return json.dumps(g.transcript(), indent=2).encode()


# -------------------------------------------------------------------- tests
@pytest.mark.parametrize("schedule", ["parallel", "sequential"])
def test_repeatable_under_random_latency(schedule):
    """Same run, five different latency patterns: byte-identical transcripts."""
    outs = {_bytes(asyncio.run(_one_run(ScriptedClient(seed), "task A", schedule))[0])
            for seed in range(5)}
    assert len(outs) == 1


def test_the_scripted_model_really_exercises_shared_state():
    g, ctx = asyncio.run(_one_run(ScriptedClient(0), "task A"))
    assert len(ctx.blackboard.data) == 5 * 3                  # 5 agents x 3 rounds
    assert "outbox.jsonl" in ctx.files and "shared.txt" in ctx.files
    # in lockstep, each agent's read at step 1 sees every agent's step-0 write
    names = [a.name for a in g.agents]
    assert all(all(f'"{n}_' in msg for n in names) for _, msg in g.communication_data[0])


def test_negative_control_naive_gather_is_not_repeatable():
    """The pre-C02 schedule (plain asyncio.gather over whole turns) depends on
    latency. If this ever stops failing, the test above has lost its power."""
    async def naive(seed):
        client = ScriptedClient(seed)
        specs = build_team(5)
        ctx = RunContext(client, [s.name for s in specs])
        g = AgentGraph(specs, topology.build_adjacency("complete", 5), ctx)
        await asyncio.gather(*(a.act("TASK: naive", 0) for a in g.agents))
        return json.dumps([a.last_response for a in g.agents])
    outs = {asyncio.run(naive(seed)) for seed in range(20)}
    assert len(outs) > 1


def test_concurrent_runs_share_no_state():
    async def both():
        client = ScriptedClient(1)      # one shared client, as in a real sweep
        return await asyncio.gather(_one_run(client, "task A"), _one_run(client, "task B"))
    (ga, ca), (gb, cb) = asyncio.run(both())
    solo_a = asyncio.run(_one_run(ScriptedClient(2), "task A"))[0]
    solo_b = asyncio.run(_one_run(ScriptedClient(3), "task B"))[0]
    assert _bytes(ga) == _bytes(solo_a)
    assert _bytes(gb) == _bytes(solo_b)
    assert ca.blackboard is not cb.blackboard and ca.files is not cb.files
    assert _bytes(ga) != _bytes(gb)


def test_no_wall_clock_in_prompts_or_transcript(monkeypatch):
    runs = []
    for fake_now in (1_000_000_000.0, 2_000_000_000.0):
        monkeypatch.setattr(time, "time", lambda t=fake_now: t)
        client = ScriptedClient(0)
        g, _ = asyncio.run(_one_run(client, "task A"))
        runs.append((_bytes(g), json.dumps(client.prompts)))
    assert runs[0] == runs[1]


def test_saved_transcript_file_is_byte_identical(tmp_path):
    paths = []
    for i in range(2):
        g, _ = asyncio.run(_one_run(ScriptedClient(i), "task A"))
        paths.append(g.save(f"r{i}", out_dir=tmp_path, started=time.time()))
    a, b = ((p / "transcript.json").read_bytes() for p in paths)
    assert a == b
    assert all((p / "meta.json").exists() for p in paths)
    with pytest.raises(FileExistsError):                    # never overwrite a run
        g.save("r0", out_dir=tmp_path)


def test_fresh_run_starts_empty_and_snapshots_carry_state_only_when_asked(tmp_path):
    g1, c1 = asyncio.run(_one_run(ScriptedClient(0), "task A", state_dir=tmp_path))
    assert sum(len(s.items) for s in c1.stores.values()) == 5 * 3   # own answers, 3 rounds
    c1.save_snapshot("after_a")

    _, c2 = asyncio.run(_one_run(ScriptedClient(0), "task B", state_dir=tmp_path))
    assert c1.blackboard.data and not set(c2.blackboard.data) & set(c1.blackboard.data)
    assert not any(it.text in {x.text for s in c1.stores.values() for x in s.items}
                   for s in c2.stores.values() for it in s.items)
    assert sum(len(s.items) for s in c2.stores.values()) == 5 * 3   # only its own

    specs = build_team(5)
    c3 = RunContext(ScriptedClient(0), [s.name for s in specs], "after_a", tmp_path)
    assert c3.blackboard.data == c1.blackboard.data
    assert c3.files == c1.files
    assert {n: len(s.items) for n, s in c3.stores.items()} == \
           {n: len(s.items) for n, s in c1.stores.items()}
    assert c3.clock() > max(e["seq"] for e in c1.blackboard.log)  # clock continues

    with pytest.raises(SnapshotError):
        c1.save_snapshot("after_a")                   # immutable
    with pytest.raises(SnapshotError):
        c1.save_snapshot(EMPTY)                       # reserved
    with pytest.raises(SnapshotError):
        RunContext(ScriptedClient(0), ["Someone"], "after_a", tmp_path)   # roster mismatch
    with pytest.raises(SnapshotError):
        RunContext(ScriptedClient(0), [s.name for s in specs], "missing", tmp_path)


def test_snapshot_refuses_a_different_embedding_space(tmp_path):
    _, c1 = asyncio.run(_one_run(ScriptedClient(0), "task A", state_dir=tmp_path))
    c1.save_snapshot("s")
    other = ScriptedClient(0)
    other.embed_space = "text-embedding-3-small"
    with pytest.raises(SnapshotError):
        RunContext(other, [s.name for s in build_team(5)], "s", tmp_path)


def test_forced_answer_after_tool_budget():
    """Turn stepping keeps the old loop's semantics: max_tool_iters calls with
    tools, then one call without tools after a budget message."""
    client = ScriptedClient(0, always_tools=True)
    specs = [AgentSpec("Solo", "tester", "test", ["blackboard_read"])]
    ctx = RunContext(client, ["Solo"])
    g = AgentGraph(specs, np.zeros((1, 1), dtype=int), ctx)
    text = asyncio.run(g.agents[0].act("TASK: loop", 0))
    assert len(client.prompts) == CFG.max_tool_iters + 1
    assert client.prompts[-1][-1]["content"].startswith("Tool budget reached")
    assert text.startswith("final after")
    assert len(g.agents[0].tool_log) == CFG.max_tool_iters


def test_sql_tool_cannot_modify_the_shared_database():
    before = json.loads(asyncio.run(REGISTRY.call("sql_query", {"sql": "SELECT COUNT(*) AS n FROM sales"})))
    out = json.loads(asyncio.run(REGISTRY.call(
        "sql_query", {"sql": "WITH x AS (SELECT 1) DELETE FROM sales"})))
    after = json.loads(asyncio.run(REGISTRY.call("sql_query", {"sql": "SELECT COUNT(*) AS n FROM sales"})))
    assert "error" in out
    assert before["rows"] == after["rows"]


def test_model_cannot_supply_the_tool_context():
    ctx = RunContext(ScriptedClient(0), ["A"])
    tctx = ToolCtx(run=ctx, agent_idx=0, agent_name="A", round_idx=0)
    out = json.loads(asyncio.run(REGISTRY.call("blackboard_write", {"key": "k", "value": "v", "ctx": "x"}, tctx)))
    assert "error" in out and ctx.blackboard.data == {}


@pytest.mark.parametrize("path", ["/etc/passwd", "../x", "a/../../x", "..", ""])
def test_workspace_paths_cannot_escape(path):
    ctx = RunContext(ScriptedClient(0), ["A"])
    tctx = ToolCtx(run=ctx, agent_idx=0, agent_name="A", round_idx=0)
    out = json.loads(asyncio.run(REGISTRY.call("write_file", {"path": path, "content": "x"}, tctx)))
    assert "error" in out and ctx.files == {}


def test_embedding_failure_is_loud_unless_fallback_is_on(tmp_path):
    class _Boom:
        class embeddings:
            @staticmethod
            async def create(**kw):
                raise ConnectionError("network down")

    def client(fallback):
        c = LLMClient(Config(api_key="unused", fake_llm=False, max_retries=1,
                             embed_fallback=fallback, cache_path=tmp_path / "c.sqlite"))
        c._client = _Boom()
        return c

    with pytest.raises(RuntimeError, match="embed_fallback is off"):
        asyncio.run(client(False).embed(["hello"]))
    c = client(True)
    assert len(asyncio.run(c.embed(["hello"]))[0]) == 256
    assert len(c.cache) == 0            # a fallback vector must never be stored
