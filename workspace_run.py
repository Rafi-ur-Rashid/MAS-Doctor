"""Run AgentDojo workspace tasks with the Track W team or the single-agent baseline (C07).

    python workspace_run.py user_task_0 user_task_18 --mode mas --topology star
    python workspace_run.py user_task_0 --mode single

One run = one user task, in a fresh environment (optionally with injections):
  mas     the 5-agent team (workspace_team.py) over the chosen topology: round 0,
          `rounds` exchange rounds, then the Coordinator's final step, whose answer
          is AgentDojo's model_output.
  single  AgentDojo's own setting: one agent, its system message, all 24 tools.
Both use the same agent loop, with AgentDojo's limit of 15 tool-calling LLM calls
per turn. Outcomes are AgentDojo's utility() (and security() when an injection task
is given), evaluated on the post-run environment. Each run record under runs/<id>/
holds transcript.json (with the tool calls and outcomes), events.jsonl, meta.json
(usage, cache statistics, manifest) and environment_post.json, from which any
outcome can be recomputed offline.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import numpy as np

import topology
from agentdojo_tools import attach, security, utility
from cache import MODES
from config import CFG
from events import event
from graph import AgentGraph
from llm import LLMClient
from run_context import EMPTY, RunContext
from workspace_team import SCRIPT, SINGLE_SCRIPT, TEAM, single_agent_spec

MAX_TOOL_ITERS = 15          # AgentDojo's ToolsExecutionLoop default
DEFAULT_ROUNDS = 3           # exchange rounds after round 0 (see EXPERIMENT_PLAN.md C07)


async def run_task(client, user_task_id: str, mode: str = "mas", topo: str = "star",
                   rounds: int = DEFAULT_ROUNDS, schedule: str = "parallel",
                   injections: dict[str, str] | None = None,
                   injection_task_id: str | None = None, state_from: str = EMPTY,
                   state_dir=None, cache_mode: str | None = None, replica: int = 0,
                   replica_from_call: int = 0, verbose: bool = False):
    if mode == "mas":
        specs, script = TEAM, SCRIPT
        adj = topology.build_adjacency(topo, len(specs))
    elif mode == "single":
        specs, script = [single_agent_spec()], SINGLE_SCRIPT
        adj, rounds = np.zeros((1, 1), dtype=int), 0
    else:
        raise ValueError(f"unknown mode {mode!r} (mas|single)")
    ctx = RunContext(client, [s.name for s in specs], state_from=state_from, state_dir=state_dir,
                     cache_mode=cache_mode, replica=replica, replica_from_call=replica_from_call,
                     toolset="workspace", max_tool_iters=MAX_TOOL_ITERS)
    ws = attach(ctx, user_task_id, injections)
    g = AgentGraph(specs, adj, ctx, script)
    task = ws.user_task.PROMPT
    await g.run(task, rounds=rounds, schedule=schedule, verbose=verbose)
    if mode == "mas":
        await g.finalize(task, idx=0, verbose=verbose)
    else:
        g.answer_from(0)

    g.outcomes = {"utility": utility(ws, g.final_answer)}
    if injection_task_id is not None:
        g.outcomes["injection_task"] = injection_task_id
        g.outcomes["security"] = security(ws, injection_task_id, g.final_answer)
    ctx.events.emit(event("outcome", user_task=user_task_id, final="final", **g.outcomes))
    return g, ctx


def save(g: AgentGraph, run_id: str, started: float, out_dir=None):
    env = g.ctx.workspace.env.model_dump_json(indent=1)
    return g.save(run_id, out_dir=out_dir, started=started,
                  extra_files={"environment_post.json": env})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("tasks", nargs="+", help="user task ids, e.g. user_task_0")
    p.add_argument("--mode", default="mas", choices=["mas", "single"])
    p.add_argument("--topology", default="star", choices=["star", "chain", "complete"])
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--schedule", default="parallel", choices=["parallel", "sequential"])
    p.add_argument("--run-prefix", default="w", help="run ids are <prefix>_<task>_<mode>...")
    p.add_argument("--cache", default=CFG.cache_mode, choices=MODES)
    p.add_argument("--budget", type=float, default=CFG.budget_usd,
                   help="hard cap in USD for this whole invocation (upper-bound pricing)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


async def main():
    args = parse_args()
    CFG.budget_usd = args.budget
    CFG.ensure_dirs()
    client = LLMClient(CFG)
    print(f"model {CFG.model} | reasoning_effort {CFG.reasoning_effort} | cache {args.cache} "
          f"| budget ${CFG.budget_usd:.2f}")
    rows = []
    for tid in args.tasks:
        started = time.time()
        g, ctx = await run_task(client, tid, args.mode, args.topology, args.rounds,
                                args.schedule, cache_mode=args.cache, verbose=args.verbose)
        tag = (f"{args.run_prefix}_{tid}_single" if args.mode == "single" else
               f"{args.run_prefix}_{tid}_{args.topology}_{args.schedule}_r{args.rounds}")
        run_dir = save(g, tag, started)
        u, st = ctx.usage, ctx.cache_stats
        rows.append({"task": tid, "utility": g.outcomes["utility"], "llm_calls": u.calls,
                     "tool_calls": len(ctx.workspace.traces), "cost_upper": round(u.cost_usd_upper, 4),
                     "hits": st.chat_hits, "misses": st.chat_misses, "truncated": st.truncated,
                     "run": str(run_dir)})
        print(json.dumps(rows[-1]))
    print(f"\nutility {sum(r['utility'] for r in rows)}/{len(rows)} | "
          f"process usage {client.usage}")


if __name__ == "__main__":
    asyncio.run(main())
