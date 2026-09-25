"""C03 checks (EXPERIMENT_PLAN.md §9.3): record-replay cache, replicas,
divergence point, budget cap. All offline.

StubOpenAI stands in for the API and samples like a real model: every call
returns a new random answer after a random delay. So a replay can only match
the recording if the cache is doing its job.
"""
import asyncio
import json
import random
import uuid
from types import SimpleNamespace

import pytest
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

import topology
from agent import AgentSpec
from cache import BudgetExceeded, CacheMiss, ResponseCache, chat_key
from config import Config
from graph import AgentGraph
from llm import LLMClient
from run_context import RunContext
from team import build_team


class StubOpenAI:
    def __init__(self, max_delay: float = 0.005, finish: str | None = None):
        self.calls = 0
        self.embed_calls = 0
        self.max_delay = max_delay
        self.finish = finish
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.embeddings = SimpleNamespace(create=self._embed)

    async def _create(self, **req):
        self.calls += 1
        await asyncio.sleep(random.random() * self.max_delay)
        msgs, tools = req["messages"], req.get("tools")
        last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
        used_tool = any(m["role"] == "tool" for m in msgs[last_user:])
        nonce = uuid.uuid4().hex[:8]                      # never repeats itself
        if tools and not used_tool:
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_{nonce}", "type": "function",
                 "function": {"name": "blackboard_write",
                              "arguments": json.dumps({"key": f"k_{nonce}", "value": nonce})}}]}
            finish = "tool_calls"
        else:
            message, finish = {"role": "assistant", "content": f"answer {nonce}"}, "stop"
        return ChatCompletion.model_validate({
            "id": f"chatcmpl-{nonce}", "object": "chat.completion", "created": 0,
            "model": req["model"],
            "choices": [{"index": 0, "finish_reason": self.finish or finish, "message": message}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
                      "completion_tokens_details": {"reasoning_tokens": 60},
                      "prompt_tokens_details": {"cached_tokens": 0}}})

    async def _embed(self, model, input):
        self.embed_calls += 1
        await asyncio.sleep(random.random() * self.max_delay)
        return CreateEmbeddingResponse.model_validate({
            "object": "list", "model": model,
            "data": [{"object": "embedding", "index": i,
                      "embedding": [random.random() for _ in range(8)]} for i in range(len(input))],
            "usage": {"prompt_tokens": 5 * len(input), "total_tokens": 5 * len(input)}})


def make(tmp_path, stub=None, **cfg):
    cfg = Config(api_key=cfg.pop("api_key", "stub"), fake_llm=False,
                 cache_path=tmp_path / "cache.sqlite", max_retries=1,
                 budget_usd=cfg.pop("budget_usd", 100.0), **cfg)
    client = LLMClient(cfg)
    stub = stub or StubOpenAI()
    if cfg.api_key:
        client._client = stub
    return client, stub


async def run_graph(client, task="task A", specs=None, **ctx_kw):
    specs = specs or build_team(3)
    ctx = RunContext(client, [s.name for s in specs], **ctx_kw)
    g = AgentGraph(specs, topology.build_adjacency("star", len(specs)), ctx)
    await g.run(task, rounds=1, schedule="parallel", verbose=False)
    await g.synthesize(task)
    return json.dumps(g.transcript(), indent=2), ctx, g


# ----------------------------------------------------------------- replay
def test_strict_replay_reproduces_a_recorded_run_with_zero_api_calls(tmp_path):
    client, stub = make(tmp_path)
    recorded, ctx1, _ = asyncio.run(run_graph(client))
    assert stub.calls > 0 and ctx1.cache_stats.chat_hits == 0

    client2, stub2 = make(tmp_path)                    # fresh process, same store
    replayed, ctx2, _ = asyncio.run(run_graph(client2, cache_mode="replay-strict"))
    assert replayed == recorded
    assert stub2.calls == 0 and stub2.embed_calls == 0
    assert ctx2.cache_stats.chat_misses == 0 and ctx2.cache_stats.first_miss_call is None
    assert ctx2.cache_stats.chat_hits == stub.calls
    assert ctx2.usage.cost_usd_upper == 0                # replays cost nothing


def test_negative_control_without_the_cache_runs_differ(tmp_path):
    client, _ = make(tmp_path)
    a = asyncio.run(run_graph(client, cache_mode="off"))[0]
    b = asyncio.run(run_graph(client, cache_mode="off"))[0]
    assert a != b


def test_strict_replay_of_a_changed_run_raises(tmp_path):
    client, _ = make(tmp_path)
    asyncio.run(run_graph(client, "task A"))
    with pytest.raises(CacheMiss, match="chat call 0"):
        asyncio.run(run_graph(client, "task B", cache_mode="replay-strict"))


def test_replay_strict_needs_no_api_key(tmp_path):
    client, _ = make(tmp_path)
    recorded = asyncio.run(run_graph(client))[0]
    keyless, _ = make(tmp_path, api_key="")
    assert asyncio.run(run_graph(keyless, cache_mode="replay-strict"))[0] == recorded
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        asyncio.run(run_graph(keyless, "task B"))


# ---------------------------------------------------------- divergence point
def test_first_miss_is_the_divergence_point(tmp_path):
    """Change only agent 2's instructions. In round 0 the three agents' first
    calls are numbered 0, 1, 2, so the paired run must diverge exactly at call 2
    and reuse the stored answers before it."""
    client, _ = make(tmp_path)
    asyncio.run(run_graph(client))
    specs = build_team(3)
    specs[2] = AgentSpec(specs[2].name, specs[2].role, specs[2].instructions + " (altered)",
                         specs[2].tools)
    _, ctx, _ = asyncio.run(run_graph(client, specs=specs))
    st = ctx.cache_stats
    assert st.first_miss_call == 2
    assert 0 not in st.missed_calls and 1 not in st.missed_calls
    assert st.chat_hits >= 2


def test_call_numbering_does_not_depend_on_latency(tmp_path):
    def numbering(seed):
        random.seed(seed)
        client, _ = make(tmp_path, StubOpenAI(max_delay=0.02))
        seen = []
        real_chat = client.chat

        async def spy(messages, tools=None, **kw):
            seen.append((kw["idx"], messages[0]["content"][:20],
                         sum(m["role"] == "assistant" for m in messages)))
            return await real_chat(messages, tools, **kw)
        client.chat = spy
        asyncio.run(run_graph(client, cache_mode="off"))
        return sorted(seen)
    assert len({json.dumps(numbering(s)) for s in range(5)}) == 1


# ------------------------------------------------------------------ replicas
def test_replica_from_call_reuses_the_prefix_and_resamples_after_it(tmp_path):
    client, _ = make(tmp_path)
    base, ctx0, _ = asyncio.run(run_graph(client))
    n = ctx0.cache_stats.chat_misses

    rep, ctx1, _ = asyncio.run(run_graph(client, replica=1, replica_from_call=3))
    st = ctx1.cache_stats
    assert st.chat_hits == 3 and st.first_miss_call == 3
    assert st.missed_calls == list(range(3, 3 + st.chat_misses))
    assert rep != base

    # a replica is itself a recording: the same settings replay it exactly
    again, ctx2, _ = asyncio.run(run_graph(client, replica=1, replica_from_call=3,
                                           cache_mode="replay-strict"))
    assert again == rep and ctx2.cache_stats.chat_misses == 0

    _, ctx3, _ = asyncio.run(run_graph(client, replica=2))           # from call 0
    assert ctx3.cache_stats.chat_hits == 0 and ctx3.cache_stats.chat_misses >= n


def test_replica_settings_are_validated(tmp_path):
    client, _ = make(tmp_path)
    with pytest.raises(ValueError):
        RunContext(client, ["A"], replica=-1)
    with pytest.raises(ValueError):
        RunContext(client, ["A"], cache_mode="record")


# ------------------------------------------------------------ store behaviour
def test_concurrent_misses_on_the_same_request_agree(tmp_path):
    """Two processes miss on one request at once: both pay, the first write
    wins, and both continue with the stored answer."""
    a, stub_a = make(tmp_path, StubOpenAI(max_delay=0.02))
    b, stub_b = make(tmp_path, StubOpenAI(max_delay=0.02))
    b._cache = ResponseCache(tmp_path / "cache.sqlite")               # separate connection
    msgs = [{"role": "user", "content": "hi"}]

    async def both():
        return await asyncio.gather(a.chat(msgs), b.chat(msgs))
    ma, mb = asyncio.run(both())
    assert stub_a.calls == 1 and stub_b.calls == 1
    assert ma.content == mb.content
    assert a.cache.get(chat_key(dict(model=a.cfg.model, messages=msgs,
                                     reasoning_effort="low", max_completion_tokens=4096), 0)) \
        ["choices"][0]["message"]["content"] == ma.content


def test_key_covers_every_request_field():
    base = {"model": "m", "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "t"}}],
            "reasoning_effort": "low", "max_completion_tokens": 10}
    variants = [dict(base, model="m2"),
                dict(base, messages=[{"role": "user", "content": "y"}]),
                dict(base, tools=[]),
                dict(base, reasoning_effort="medium"),
                dict(base, max_completion_tokens=11)]
    keys = {chat_key(v, 0) for v in variants} | {chat_key(base, 0), chat_key(base, 1)}
    assert len(keys) == len(variants) + 2
    assert chat_key(dict(reversed(list(base.items()))), 0) == chat_key(base, 0)


def test_embeddings_are_cached_and_strict_replay_refuses_new_text(tmp_path):
    client, stub = make(tmp_path)
    v1 = asyncio.run(client.embed(["a", "b"]))
    v2 = asyncio.run(client.embed(["b", "a"]))
    assert stub.embed_calls == 1 and v2 == [v1[1], v1[0]]
    with pytest.raises(CacheMiss):
        asyncio.run(client.embed(["c"], mode="replay-strict"))


def test_truncated_responses_are_counted(tmp_path):
    client, _ = make(tmp_path, StubOpenAI(finish="length"))
    _, ctx, _ = asyncio.run(run_graph(client))
    assert ctx.cache_stats.truncated == ctx.cache_stats.chat_misses > 0


# -------------------------------------------------------------------- budget
def test_budget_cap_stops_further_api_calls(tmp_path):
    client, stub = make(tmp_path, budget_usd=1e-9)
    asyncio.run(client.chat([{"role": "user", "content": "one"}]))
    # 1000 prompt tokens at $0.25/M + 100 completion at $2.00/M
    assert client.usage.cost_usd_upper == pytest.approx(0.00045)
    with pytest.raises(BudgetExceeded):
        asyncio.run(client.chat([{"role": "user", "content": "two"}]))
    assert stub.calls == 1
    # the cap never blocks what is already stored
    asyncio.run(client.chat([{"role": "user", "content": "one"}]))
    assert stub.calls == 1


def test_a_model_without_a_price_is_refused(tmp_path):
    with pytest.raises(KeyError, match="no price"):
        LLMClient(Config(api_key="x", fake_llm=False, model="gpt-unpriced",
                         cache_path=tmp_path / "c.sqlite"))


# ------------------------------------------------------------------- record
def test_saved_meta_carries_cache_stats_and_transcript_hash(tmp_path):
    import hashlib
    client, _ = make(tmp_path)
    _, ctx, g = asyncio.run(run_graph(client))
    run_dir = g.save("r", out_dir=tmp_path / "runs")
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["cache"]["mode"] == "replay-or-record"
    assert meta["cache"]["chat_misses"] == ctx.cache_stats.chat_misses
    assert meta["transcript_sha256"] == hashlib.sha256(
        (run_dir / "transcript.json").read_bytes()).hexdigest()
