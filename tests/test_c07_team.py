"""C07 checks (EXPERIMENT_PLAN.md §9.3): the Track W team, the single-agent
baseline and the runner. All offline; the live check is run by hand (PROGRESS.md).

TeamStub plays the model through the real LLMClient: in round 0 the calendar
specialist (or the single agent) looks up the event user_task_0 asks about; later
it restates what it found, the others pass on what their neighbours said, and the
Coordinator's final answer draws on its whole context. So AgentDojo's utility() can only pass if the facts travel
tool -> specialist -> Coordinator -> final answer.
"""
import asyncio
import json
import re
import uuid

import pytest
from openai.types.chat import ChatCompletion

from agentdojo_tools import ANNOTATIONS, workspace_suite
from events import load, verify_records
from test_c03_cache import StubOpenAI, make
from workspace_run import run_task, save
from workspace_team import (CALENDAR, DRIVE, EMAIL, READ_ONLY, SHARED_LOOKUPS, TEAM,
                            agentdojo_system_message, single_agent_spec)

TASK = "user_task_0"      # "Who else is invited to the 'Networking event' on May 26th? ..."


class TeamStub(StubOpenAI):
    def __init__(self, rogue: str | None = None):
        super().__init__()
        self.rogue = rogue          # an agent that tries a tool outside its ACL

    async def _create(self, **req):
        self.calls += 1
        msgs, tools = req["messages"], req.get("tools")
        system = msgs[0]["content"]
        m = re.match(r"You are agent_\d+ \((\w+)\)", system)
        name = m.group(1) if m else "Assistant"
        last_user = max(i for i, x in enumerate(msgs) if x["role"] == "user")
        since = msgs[last_user:]
        results = [x["content"] for x in since if x["role"] == "tool"]
        first_turn = sum(1 for x in msgs if x["role"] == "user") == 1
        nonce = uuid.uuid4().hex[:8]                 # a real model never repeats itself
        call = None
        if tools and not results and first_turn:
            if name in ("CalendarAgent", "Assistant"):
                call = ("search_calendar_events", {"query": "Networking event", "date": "2024-05-26"})
            elif name == self.rogue:
                call = ("create_calendar_event", {"title": "x", "start_time": "2024-05-20 10:00",
                                                  "end_time": "2024-05-20 11:00"})
        if call:
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_{nonce}", "type": "function",
                 "function": {"name": call[0], "arguments": json.dumps(call[1])}}]}
            finish = "tool_calls"
        elif results:
            message, finish = {"role": "assistant", "content": f"Found: {results[-1]}"}, "stop"
        elif "The team has finished" in msgs[last_user]["content"]:
            # a real model answers from its whole context, not just the last message
            heard = "\n".join(x["content"] for x in msgs if x["role"] == "user")
            message, finish = {"role": "assistant", "content": heard}, "stop"
        elif results := [x["content"] for x in msgs if x["role"] == "tool"]:
            # restates what it found in an earlier round
            message, finish = {"role": "assistant", "content": f"Earlier: {results[-1]} ({nonce})"}, "stop"
        else:
            # passes on what its neighbours told it
            relay = msgs[last_user]["content"].split("connected to you:", 1)[-1]
            message, finish = {"role": "assistant", "content": f"{name} heard: {relay} ({nonce})"}, "stop"
        return ChatCompletion.model_validate({
            "id": f"chatcmpl-{nonce}", "object": "chat.completion", "created": 0,
            "model": req["model"],
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}})


# ------------------------------------------------------------------ roster
def test_acls_follow_the_services():
    by = {s.name: s for s in TEAM}
    assert [s.name for s in TEAM] == ["Coordinator", "EmailAgent", "CalendarAgent",
                                      "DriveAgent", "Reviewer"]
    assert by["Coordinator"].tools == []
    assert set(by["EmailAgent"].tools) == set(EMAIL) | set(SHARED_LOOKUPS)
    assert set(by["CalendarAgent"].tools) == set(CALENDAR) | set(SHARED_LOOKUPS)
    assert set(by["DriveAgent"].tools) == set(DRIVE) | set(SHARED_LOOKUPS)
    assert all(ANNOTATIONS[t].side_effect == "read_only" for t in SHARED_LOOKUPS)
    assert set(by["Reviewer"].tools) == set(READ_ONLY)
    assert all(ANNOTATIONS[t].side_effect == "read_only" for t in by["Reviewer"].tools)
    assert "get_unread_emails" not in by["Reviewer"].tools        # it marks emails read
    # every tool that changes state belongs to exactly one specialist
    for tool, a in ANNOTATIONS.items():
        holders = [s.name for s in TEAM if tool in s.tools]
        if a.side_effect != "read_only":
            assert len(holders) == 1, (tool, holders)
        else:
            assert holders, tool
    assert sorted(single_agent_spec().tools) == sorted(ANNOTATIONS)


def test_single_agent_sees_exactly_agentdojos_prompts(tmp_path):
    client, _ = make(tmp_path, stub=TeamStub())
    g, ctx = asyncio.run(run_task(client, TASK, mode="single"))
    first = next(r for r in ctx.events.records if r.get("ev") == "llm_call")
    from events import rebuild_message
    msgs = [rebuild_message(m, ctx.events.artifacts) for m in first["context"]]
    assert msgs[0] == {"role": "system", "content": agentdojo_system_message()}
    assert msgs[1] == {"role": "user", "content": workspace_suite().user_tasks[TASK].PROMPT}
    assert len(msgs) == 2 and len(first["tools"]) == 24
    assert g.outcomes == {"utility": True}


# ------------------------------------------------------------- team runs
@pytest.mark.parametrize("topo", ["star", "chain", "complete"])
def test_team_run_relays_the_facts_to_the_final_answer(tmp_path, topo):
    client, stub = make(tmp_path, stub=TeamStub())
    g, ctx = asyncio.run(run_task(client, TASK, mode="mas", topo=topo, rounds=3))
    assert len(g.communication_data) == 4                    # round 0 + 3 exchange rounds
    final = ctx.events.artifacts["final"]
    assert final["derived_from"] == ["resp:0:4"]              # the Coordinator's final step
    assert g.final_answer == ctx.events.artifacts["resp:0:4"]["content"]
    assert [c.function for c in ctx.workspace.traces] == ["search_calendar_events"]
    # the facts reach the Coordinator in round 1 on star and complete, but only by the
    # final step on a chain whose calendar agent is two hops away: they still arrive
    assert g.outcomes == {"utility": True}
    outcome = [r for r in ctx.events.records if r.get("ev") == "outcome"]
    assert outcome == [{**outcome[0], "user_task": TASK, "utility": True}]
    assert verify_records(ctx.events.records, cache=client.cache)["llm_calls"] == stub.calls


def test_an_unreachable_answer_fails_utility(tmp_path):
    """Negative control: with no exchange rounds on a chain, the calendar agent's
    finding never reaches the Coordinator, and utility() says so."""
    client, _ = make(tmp_path, stub=TeamStub())
    g, _ = asyncio.run(run_task(client, TASK, mode="mas", topo="chain", rounds=0))
    assert g.outcomes == {"utility": False}


def test_a_tool_outside_the_acl_is_refused_and_never_runs(tmp_path):
    client, _ = make(tmp_path, stub=TeamStub(rogue="EmailAgent"))
    g, ctx = asyncio.run(run_task(client, TASK, mode="mas", rounds=1))
    d = [r for r in ctx.events.records if r.get("ev") == "tool_dispatch"
         and r["tool"] == "create_calendar_event"]
    assert len(d) == 1 and d[0]["agent"] == "EmailAgent" and d[0]["allowed"] is False
    res = next(r for r in ctx.events.records if r.get("ev") == "tool_result"
               and r["tool"] == "create_calendar_event")
    assert ctx.events.artifacts[res["result"]]["content"] == \
        "Invalid tool create_calendar_event provided."            # AgentDojo's wording
    assert res["observations"] == []
    assert [c.function for c in ctx.workspace.traces] == ["search_calendar_events"]
    assert len(ctx.workspace.env.calendar.events) == len(ctx.workspace.pre_environment.calendar.events)


def test_run_record_and_strict_replay(tmp_path):
    client, stub = make(tmp_path, stub=TeamStub())
    g1, c1 = asyncio.run(run_task(client, TASK, mode="mas"))
    run_dir = save(g1, "r1", 0.0, out_dir=tmp_path / "runs")
    t = json.loads((run_dir / "transcript.json").read_text())
    assert t["workspace"]["user_task"] == TASK and t["workspace"]["outcomes"] == {"utility": True}
    assert t["workspace"]["tool_calls"][0]["function"] == "search_calendar_events"
    meta = json.loads((run_dir / "meta.json").read_text())
    assert set(meta["files"]) == {"environment_post.json"}
    assert verify_records(load(run_dir / "events.jsonl"), cache=client.cache)["cache_keys_checked"] \
        == stub.calls
    start = next(r for r in c1.events.records if r.get("ev") == "run_start")
    assert start["toolset"] == "workspace" and start["workspace"]["user_task"] == TASK

    client2, stub2 = make(tmp_path, stub=TeamStub())
    g2, c2 = asyncio.run(run_task(client2, TASK, mode="mas", cache_mode="replay-strict"))
    assert stub2.calls == 0 and c2.cache_stats.chat_misses == 0
    assert json.dumps(g2.transcript()) == json.dumps(g1.transcript())
    assert c2.workspace.env.model_dump_json() == c1.workspace.env.model_dump_json()


def test_a_stray_get_unread_emails_fails_most_tasks_hence_the_email_prompt():
    """get_unread_emails marks email read. After a task's own ground truth, one
    extra call fails 37 of 40 utility checks (they require the rest of the
    environment unchanged), which is why the EmailAgent's instructions reserve it
    for tasks about unread email (seen live in C07)."""
    from agentdojo.functions_runtime import FunctionCall
    from test_c06_agentdojo import USER_TASKS, run_calls
    from agentdojo_tools import utility
    suite = workspace_suite()
    failed = []
    for t in USER_TASKS:
        task = suite.user_tasks[t]
        _, ws, _ = run_calls(t, {}, lambda env: task.ground_truth(env)
                             + [FunctionCall(function="get_unread_emails", args={})])
        failed += [] if utility(ws, task.GROUND_TRUTH_OUTPUT) else [t]
    assert len(failed) == 37
    email_agent = next(s for s in TEAM if s.name == "EmailAgent")
    assert "get_unread_emails marks every unread email as read" in email_agent.instructions
