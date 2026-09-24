# Local Multi-Agent System

A graph-structured MAS in the shape of XG-Guard's setup, but with the parts that were
simulated there made real: tools actually execute, memory actually persists and is
retrieved by embedding similarity, and agents share a live blackboard.

No framework — plain `openai` + `numpy`, ~900 lines. Nothing is hidden behind a
LangGraph/AutoGen abstraction, so every prompt and every message hop is inspectable.

## Layout

| file | what it holds |
|---|---|
| `config.py`   | model, paths, budgets. Everything tunable in one dataclass. |
| `llm.py`      | async OpenAI wrapper: retries, concurrency cap, usage accounting, offline fake mode |
| `tools.py`    | the tool registry + 8 executing tools |
| `memory.py`   | working / episodic (vector) / shared blackboard |
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
```

`--fresh` wipes persisted memory and the blackboard first; without it, agents carry
episodic memory across runs, which is usually what you want on the second run.

## The three layers, concretely

**Tools** (`tools.py`) — all real except one. `calculator` evaluates via a whitelisted
AST walk. `sql_query` runs against a seeded in-memory sqlite (employees / products /
sales), SELECT-only, single statement. `kb_search` retrieves from the markdown corpus in
`knowledge_base/`. `read_file` / `write_file` / `list_files` are sandboxed to
`workspace/` with an escape check. `blackboard_read` / `blackboard_write` are the shared
channel. `send_email` is the one deliberate simulation: it appends to
`workspace/outbox.jsonl` so the side effect is observable without leaving the machine.

Dispatch is OpenAI native function calling, and results are fed back as `role: "tool"`
messages, looping up to `CFG.max_tool_iters` times before the agent is forced to answer.

**Memory** (`memory.py`) — three distinct things:
- *Working*: the message list sent to the model, windowed to the last `working_memory_turns`
  exchanges, never slicing a `tool` result away from its call.
- *Episodic*: per-agent `VectorStore`, embedded with `text-embedding-3-small`, persisted to
  `state/agent_N_episodic.json`, retrieved by cosine similarity and prepended to each turn's
  prompt. Falls back to hashed bag-of-words embeddings if the endpoint errors, so a
  network problem degrades retrieval instead of killing the run.
- *Blackboard*: shared key/value store with an append-only authored log, persisted to
  `state/blackboard.json`. This is how agents that aren't graph neighbours reach each other.

**Topology** (`topology.py`) — `adj[i][j] == 1` means i's messages reach j, so column j is
agent j's in-neighbours. Same convention as XG-Guard, so transcripts are comparable.

## Scheduling

`--schedule parallel` (default) runs every agent at once against a *snapshot* of the
previous round. Fast, and it's what XG-Guard does. The catch: an agent cannot see work its
peers do in the same round, so a downstream role (the Writer) can read an empty blackboard
in round 0 and have nothing to write.

`--schedule sequential` runs agents in index order within a round, each seeing whatever
its neighbours already produced this round. Slower — no parallelism — but consumer roles
actually have something to consume.

## Output

Each run writes `runs/run_<timestamp>.json` containing the task, `adj_matrix`,
`system_prompts`, and `communication_data` shaped as `[[[agent_idx, text], ...], ...]` —
one list per round. That is deliberately XG-Guard's transcript schema, so these runs can
be fed to a graph-anomaly detector later without reshaping. Also included: every tool call
with arguments and truncated results, the blackboard and its authored log, the synthesis,
and token usage.

## Adding to it

A tool is one decorated function:

```python
@REGISTRY.register("my_tool", "What it does.",
    {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]})
def my_tool(x: str):
    return {"result": ...}          # dict or str; dicts are JSON-encoded
```

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
