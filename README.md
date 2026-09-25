# Local Multi-Agent System

A graph-structured MAS in the shape of XG-Guard's setup, but with the parts that were
simulated there made real: tools actually execute, memory is stored and retrieved by
embedding similarity, and agents share a live blackboard. Every run is isolated and
repeatable: it starts from a named state snapshot and touches no global state.

No framework — plain `openai` + `numpy`, ~900 lines. Nothing is hidden behind a
LangGraph/AutoGen abstraction, so every prompt and every message hop is inspectable.

## Layout

| file | what it holds |
|---|---|
| `config.py`   | model, paths, budgets. Everything tunable in one dataclass. |
| `llm.py`      | async OpenAI wrapper: retries, concurrency cap, usage accounting, offline fake mode |
| `tools.py`    | the tool registry + 8 executing tools |
| `memory.py`   | working / episodic (vector) / shared blackboard |
| `run_context.py` | one run's mutable state + named state snapshots |
| `manifest.py` | versions, models and code hash for every run record |
| `cache.py`    | record-replay store for model responses; budget cost bounds |
| `events.py`   | event log: artifacts, recipes, events, and the verifier |
| `agent.py`    | one agent: persona, tool subset, tool-calling loop |
| `topology.py` | chain, ring, star, tree, complete, random |
| `graph.py`    | round-based message passing + synthesis + transcript |
| `team.py`     | the 5-agent roster |
| `run.py`      | CLI |

## Quick start

```bash
python run.py --fake-llm                 # offline smoke test, zero API cost
python run.py                            # default: 5 agents, star, 2 rounds, gpt-5-mini-2025-08-07
python run.py --manifest                 # pinned versions, models, code hash
python run.py --topology random --sparsity 0.5 --seed 7 --rounds 3
python run.py --schedule sequential      # downstream roles see this round's work
python run.py --task "Which region is underperforming and why?"
python run.py --state-save day1          # keep this run's end state as snapshot 'day1'
python run.py --state-from day1          # start the next run from it
```

Every run starts from the `empty` state unless `--state-from` names a snapshot. Nothing
carries over between runs implicitly. Snapshots live in `state/snapshots/<name>.json`, hold
episodic memory, the blackboard and workspace files, and are never overwritten.

## The three layers, concretely

**Tools** (`tools.py`) — all real except one. `calculator` evaluates via a whitelisted
AST walk. `sql_query` runs against a seeded in-memory sqlite (employees / products /
sales), SELECT-only, single statement. `kb_search` retrieves from the markdown corpus in
`knowledge_base/`. The database is opened with `PRAGMA query_only`, because the SELECT
check alone lets `WITH ... DELETE` through. `read_file` / `write_file` / `list_files`
work on the run's own in-memory workspace, with a path-escape check.
`blackboard_read` / `blackboard_write` are the shared channel. `send_email` is the one
deliberate simulation: it appends to `outbox.jsonl` in the run's workspace, so the side
effect is observable without leaving the machine.

Tools that touch run state are registered with `needs_ctx=True` and receive a `ToolCtx`
(run, agent, round) from the runtime. The model cannot supply or override it.

Dispatch is OpenAI native function calling, and results are fed back as `role: "tool"`
messages, looping up to `CFG.max_tool_iters` times before the agent is forced to answer.

**Memory** (`memory.py`) — three distinct things:
- *Working*: the message list sent to the model, windowed to the last `working_memory_turns`
  exchanges, never slicing a `tool` result away from its call.
- *Episodic*: per-agent `VectorStore`, embedded with `text-embedding-3-small`, retrieved
  by cosine similarity and prepended to each turn's prompt. If the embedding endpoint
  errors, the run stops: hashed vectors live in a different space, so a silent fallback
  would change retrieval mid-experiment. `MAS_EMBED_FALLBACK=1` allows it for demos.
- *Blackboard*: shared key/value store with an append-only authored log. This is how
  agents that aren't graph neighbours reach each other.

Ordering uses the run's logical clock (`seq`), never wall-clock time.

**Topology** (`topology.py`) — `adj[i][j] == 1` means i's messages reach j, so column j is
agent j's in-neighbours. Same convention as XG-Guard, so transcripts are comparable.

## Scheduling

`--schedule parallel` (default) runs every agent against a *snapshot* of the previous
round's messages, which is what XG-Guard does. Within a round the agents move in
lockstep: all of them make their next LLM call concurrently, then their tool calls run one
agent at a time in index order. Shared state therefore changes in the same order however
fast each API call returns, so the run is repeatable. A tool effect is visible to other
agents from the next step on. The catch: an agent does not see its peers' *messages* from
the same round, so a downstream role (the Writer) can have little to write in round 0.

`--schedule sequential` runs agents in index order within a round, each seeing whatever
its neighbours already produced this round. Slower — no parallelism — but consumer roles
actually have something to consume.

## Record and replay (`cache.py`)

gpt-5-mini cannot be made to repeat itself, so every model and embedding response is
stored in `cache/llm_cache.sqlite` under a fingerprint of the exact request (model
snapshot, messages, tools, sampling settings, replica). Running the same run again reuses
the stored responses: same transcript, zero API calls, zero cost.

```bash
python run.py --run-id a                          # records (default: replay-or-record)
python run.py --run-id b --cache replay-strict    # replays a; errors instead of calling the API
python run.py --replica 1                         # a fresh sample of the same run
python run.py --replica 1 --replica-from-call 12  # reuse calls 0-11, resample from call 12
python run.py --budget 0.50                       # stop calling the API past $0.50 (upper bound)
```

For a paired run (clean vs. attacked), the attacked run reuses every stored response
until the first request the attack changed. `meta.json` records that call's index as
`cache.first_miss_call`, the divergence point. Replays need no API key. The spending cap
counts only real API calls, at the list prices in `config.PRICES_PER_1M`, ignoring
OpenAI's cached-input discount, so it overestimates spend. A model with no price there
cannot make API calls.

## Event log (`events.py`)

Each run also writes `runs/<run_id>/events.jsonl`: one record per line, either an
**artifact** (a piece of content with an id) or an **event** (something that happened).

| Artifact id | What it is |
|---|---|
| `task` | the user task |
| `sys:<agent>` | an agent's system prompt |
| `m:<agent>:<n>` | the n-th message in an agent's context, stored as a *recipe* |
| `out:<call>` | a model output (text and tool calls) |
| `tr:<call>.<k>` | the result of the k-th tool call in that output |
| `resp:<agent>:<round>` | an agent's final response in a round (what neighbours receive) |
| `w<seq>` | an episodic memory item (its write id) |
| `bb<seq>` | a blackboard value; the moderator's prompt also references its `key` field |
| `file:g<gen>:<call>.<k>` | a workspace file version (`gen` = snapshot generation) |
| `final` | the moderator's synthesis |

Events: `run_start`, `prompt_delivery` (registered vs delivered prompt hash), `llm_call`
(the message ids that formed the context, a hash of what was sent, the cache key, the
output id), `tool_dispatch`, `tool_result` (result id plus the artifact ids the tool read
and wrote), `mem_read` (candidates, returned write ids, similarities, top-1/top-2 margin),
`mem_write`, `msg_send`, `msg_recv`.

A recipe is a list of parts: runtime-written literal text, or a reference to an artifact.
So the log shows exactly which pieces went into every prompt and which text the runtime
itself wrote; model-, tool- and memory-authored text only ever appears as a reference.
`python events.py runs/<run_id>` rebuilds every prompt from the recipes, checks it against
the hash of what was sent, rebuilds each request's cache key, and checks the key is in the
store. The same run yields a byte-identical log.

## Output

Each run writes `runs/<run_id>/transcript.json`, `events.jsonl` (above) and `meta.json`.
`transcript.json` contains no wall-clock time, so the same run from the same state gives a
byte-identical file. `meta.json` holds timings, token usage and cost, cache statistics,
the transcript's SHA-256, and the manifest.
The transcript contains the task, `adj_matrix`,
`system_prompts`, and `communication_data` shaped as `[[[agent_idx, text], ...], ...]` —
one list per round. That is deliberately XG-Guard's transcript schema, so these runs can
be fed to a graph-anomaly detector later without reshaping. Also included: every tool call
with arguments and truncated results, the blackboard and its authored log, the run's
workspace files, and the synthesis.

## Adding to it

A tool is one decorated function:

```python
@REGISTRY.register("my_tool", "What it does.",
    {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]})
def my_tool(x: str):
    return {"result": ...}          # dict or str; dicts are JSON-encoded
```

A tool that reads or changes run state adds `needs_ctx=True` and takes the context first:
`def my_tool(ctx: ToolCtx, x: str)`, then uses `ctx.run.blackboard`, `ctx.run.files`,
`ctx.agent_name`. Never keep run state in a module-level variable.

Then list its name in an `AgentSpec.tools` in `team.py`. Agents only ever see the tools
their spec names — that asymmetry is what forces them to actually talk to each other.

## Known limits

- Round count is fixed; there is no convergence check or early stop.
- The synthesizer is a single moderator call, not a vote. Swap in majority voting over
  extracted answers if you want XG-Guard's evaluation style.
- Episodic memory grows unboundedly and is scanned linearly. Fine at transcript scale,
  not at 10k items — bring in an ANN index past that.
- `sql_query` blocks non-SELECT by prefix check, which is enough for a trusted local
  sandbox and not enough for untrusted input.
