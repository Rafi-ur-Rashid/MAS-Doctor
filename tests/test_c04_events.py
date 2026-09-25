"""C04 checks (EXPERIMENT_PLAN.md §9.3): the event log. All offline.

The central property: every message the model received can be rebuilt, exactly,
from the log's recipes, and every piece of agent-, tool- or memory-authored text
in a prompt is a reference to a registered artifact, never runtime literal text.
"""
import asyncio
import json

import numpy as np
import pytest

import topology
from agent import AgentSpec
from events import load, rebuild_message, verify_records
from graph import AgentGraph
from run_context import RunContext
from team import build_team
from tools import REGISTRY
from test_c02_isolation import ScriptedClient
from test_c03_cache import make, run_graph


async def scripted_run(client, task="task A", schedule="parallel", rounds=2, topo="complete",
                       state_from="empty", state_dir=None, specs=None):
    specs = specs or build_team(5)
    ctx = RunContext(client, [s.name for s in specs], state_from=state_from, state_dir=state_dir)
    g = AgentGraph(specs, topology.build_adjacency(topo, len(specs)), ctx)
    await g.run(task, rounds=rounds, schedule=schedule, verbose=False)
    await g.synthesize(task)
    return g, ctx


def log_bytes(ctx) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ctx.events.records).encode()


def arts(ctx) -> dict:
    return ctx.events.artifacts


def evs(ctx, name) -> list:
    return [r for r in ctx.events.records if r.get("ev") == name]


# ------------------------------------------------------------- determinism
@pytest.mark.parametrize("schedule", ["parallel", "sequential"])
def test_event_log_is_byte_identical_under_random_latency(schedule):
    outs = {log_bytes(asyncio.run(scripted_run(ScriptedClient(seed), schedule=schedule))[1])
            for seed in range(5)}
    assert len(outs) == 1


# ----------------------------------------------------------- exact rebuild
@pytest.mark.parametrize("schedule", ["parallel", "sequential"])
def test_every_prompt_rebuilds_exactly_what_the_model_received(schedule):
    client = ScriptedClient(0)
    _, ctx = asyncio.run(scripted_run(client, schedule=schedule))
    report = verify_records(ctx.events.records)
    calls = sorted(evs(ctx, "llm_call"), key=lambda c: c["call_idx"])
    assert report["llm_calls"] == len(calls) == len(client.prompts)
    for c in calls:          # the scripted model saw call i as client.prompts[i]
        rebuilt = [rebuild_message(m, arts(ctx)) for m in c["context"]]
        assert json.dumps(rebuilt, sort_keys=True) == json.dumps(client.prompts[c["call_idx"]],
                                                                 sort_keys=True)


def test_verifier_catches_a_tampered_or_missing_artifact():
    _, ctx = asyncio.run(scripted_run(ScriptedClient(0)))
    records = json.loads(json.dumps(ctx.events.records))
    victim = next(r for r in records if r["type"] == "artifact" and r["kind"] == "response"
                  and r["content"])
    victim["content"] += " (edited)"
    with pytest.raises(AssertionError, match="rebuilt messages differ"):
        verify_records(records)
    records = [r for r in json.loads(json.dumps(ctx.events.records))
               if not (r["type"] == "artifact" and r["id"] == "task")]
    with pytest.raises(AssertionError, match="unresolved ref task"):
        verify_records(records)


def test_no_agent_or_tool_text_hides_in_runtime_literals():
    """Provenance soundness: text produced by a model, a tool or memory reaches a
    prompt only as a reference, so a literal part never contains it."""
    _, ctx = asyncio.run(scripted_run(ScriptedClient(0)))
    produced = [a["content"] for a in arts(ctx).values()
                if a["kind"] in ("response", "tool_result", "llm_output", "memory_item",
                                 "blackboard_value", "file")
                and isinstance(a["content"], str) and len(a["content"]) >= 12]
    assert produced
    literals = [p["text"] for a in arts(ctx).values() if a["kind"] == "message"
                for p in (a["parts"] or []) if "text" in p]
    assert not [x for x in produced for lit in literals if x in lit]


def test_round_prompts_reference_neighbour_responses_and_log_messages():
    _, ctx = asyncio.run(scripted_run(ScriptedClient(0), topo="star", rounds=1))
    recv, send = evs(ctx, "msg_recv"), evs(ctx, "msg_send")
    # star over 5: hub 0 hears 4 leaves, each leaf hears the hub
    assert len(recv) == 8 and all(r["round"] == 1 for r in recv)
    assert {(r["sender_idx"], r["to_idx"]) for r in recv} == \
        {(j, 0) for j in range(1, 5)} | {(0, j) for j in range(1, 5)}
    assert len(send) == 10 and next(s for s in send if s["sender_idx"] == 0)["to"] == [1, 2, 3, 4]
    for r in recv:
        assert r["response"] == f"resp:{r['sender_idx']}:0"
        user_msgs = [a for a in arts(ctx).values() if a["kind"] == "message"
                     and a["id"].startswith(f"m:{r['to_idx']}:") and a["role"] == "user"]
        assert any({"ref": r["response"]} in (m["parts"] or []) for m in user_msgs)


# ------------------------------------------------------- tools and memory
def test_tool_events_record_dispatch_result_reads_and_writes():
    _, ctx = asyncio.run(scripted_run(ScriptedClient(0), rounds=0))
    disp, res = evs(ctx, "tool_dispatch"), evs(ctx, "tool_result")
    assert len(disp) == len(res) > 0
    writes = [w for r in res if r["tool"] == "blackboard_write" for w in r["writes"]]
    assert writes and all(arts(ctx)[w]["kind"] == "blackboard_value" for w in writes)
    reads_file = [r for r in res if r["tool"] == "write_file"]
    assert reads_file and arts(ctx)[reads_file[0]["writes"][0]]["path"] == "shared.txt"
    for r in res:
        assert arts(ctx)[r["result"]]["kind"] == "tool_result"


def test_moderator_references_blackboard_keys_and_values():
    g, ctx = asyncio.run(scripted_run(ScriptedClient(0), rounds=0))
    mod_user = arts(ctx)["m:mod:1"]
    key_refs = [p for p in mod_user["parts"] if p.get("field") == "key"]
    assert len(key_refs) == len(g.blackboard.data)
    assert arts(ctx)["final"]["derived_from"][0].startswith("out:")


def test_memory_reads_and_writes_are_logged_with_ids(tmp_path):
    _, c1 = asyncio.run(scripted_run(ScriptedClient(0), state_dir=tmp_path))
    c1.save_snapshot("mem")
    writes = evs(c1, "mem_write")
    assert len(writes) == 15 and all(arts(c1)[w["write_id"]]["kind"] == "memory_item" for w in writes)
    assert all(w["derived_from"][0].startswith("resp:") for w in writes)

    _, c2 = asyncio.run(scripted_run(ScriptedClient(0), task="task A", state_from="mem",
                                     state_dir=tmp_path))
    loaded = [a for a in arts(c2).values() if a["producer"] == "snapshot"]
    assert {a["id"] for a in loaded if a["kind"] == "memory_item"} == {w["write_id"] for w in writes}
    reads = [r for r in evs(c2, "mem_read") if r["returned"]]
    assert reads, "same task, same memories: some recall must fire"
    for r in reads:
        assert r["n_candidates"] >= len(r["returned"]) and r["top"][0]["write_id"] == r["returned"][0]
        assert r["margin"] is None or r["margin"] >= 0
    verify_records(c2.events.records)          # loaded refs resolve


def test_prompt_delivery_records_registered_and_delivered_hashes():
    client = ScriptedClient(0)
    specs = [AgentSpec("Solo", "tester", "test", ["blackboard_read"])]
    ctx = RunContext(client, ["Solo"])
    AgentGraph(specs, np.zeros((1, 1), dtype=int), ctx)
    d = evs(ctx, "prompt_delivery")[0]
    assert d["registered_sha256"] == d["delivered_sha256"]

    from agent import Agent
    ctx2 = RunContext(client, ["Solo"])
    Agent(0, specs[0], ctx2, peers=[], registered_prompt="something else")
    d2 = evs(ctx2, "prompt_delivery")[0]
    assert d2["registered_sha256"] != d2["delivered_sha256"]


def test_every_registered_tool_is_bound_to_its_own_function():
    """Guards against a decorator ending up on the wrong function (it happened)."""
    for name, tool in REGISTRY._tools.items():
        assert tool.fn.__name__ == name


# --------------------------------------------------------- real client path
def test_cache_keys_rebuild_from_the_log(tmp_path):
    """With the real LLMClient (stub transport): every call's request, rebuilt
    from the log, has the recorded cache key, and that key is in the store."""
    client, stub = make(tmp_path)
    _, ctx, g = asyncio.run(run_graph(client))
    report = verify_records(ctx.events.records, cache=client.cache)
    assert report["cache_keys_checked"] == report["llm_calls"] == stub.calls

    run_dir = g.save("r", out_dir=tmp_path / "runs")
    assert verify_records(load(run_dir / "events.jsonl"), cache=client.cache) == report


def test_ids_stay_unique_across_a_chain_of_runs_through_snapshots(tmp_path):
    """Content that persists (memory, blackboard, files) keeps its id in the next
    run; ids minted in later runs must never collide with it."""
    state = "empty"
    for gen in range(1, 4):
        _, ctx = asyncio.run(scripted_run(ScriptedClient(0), task=f"task {gen}",
                                          state_from=state, state_dir=tmp_path, rounds=0))
        assert ctx.generation == gen
        verify_records(ctx.events.records)       # also asserts no id is registered twice
        state = f"gen{gen}"
        ctx.save_snapshot(state)
    assert any(i.startswith("file:g1:") for i in ctx.file_ids.values()) is False  # overwritten
    assert all(i.startswith("file:g3:") for i in ctx.file_ids.values())
