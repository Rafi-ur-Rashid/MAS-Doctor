#!/usr/bin/env python3
"""CLI entry point for the multi-agent system.

Examples
--------
  python run.py --fake-llm                       # free offline smoke test
  python run.py --task "..." --topology star --rounds 2
  python run.py --agents 5 --topology random --sparsity 0.5 --seed 7
"""
from __future__ import annotations

import argparse
import asyncio
import time

from config import CFG
import topology
from llm import LLMClient
from graph import AgentGraph
from run_context import EMPTY, RunContext, SnapshotError, snapshot_path
from team import build_team

DEFAULT_TASK = (
    "Which product line should Northwind Robotics prioritise for 2026 investment? "
    "Back the recommendation with 2025 sales figures from the database and with what the "
    "knowledge base says about margin and strategy. Give one clear recommendation."
)


def parse_args():
    p = argparse.ArgumentParser(description="Run the multi-agent system.")
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--agents", type=int, default=5)
    p.add_argument("--topology", default="star", choices=list(topology.KINDS))
    p.add_argument("--sparsity", type=float, default=0.5, help="edge density for --topology random")
    p.add_argument("--directed", action="store_true", help="one-way edges where the topology allows")
    p.add_argument("--rounds", type=int, default=2, help="exchange rounds after round 0")
    p.add_argument("--schedule", default="parallel", choices=["parallel", "sequential"],
                   help="within a round: all agents at once, or in index order")
    p.add_argument("--model", default=CFG.model)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fake-llm", action="store_true", help="offline stub: no API calls, no cost")
    p.add_argument("--state-from", default=EMPTY,
                   help="named state snapshot to start from (memory, blackboard, files); "
                        f"default '{EMPTY}'. Nothing carries over between runs otherwise.")
    p.add_argument("--state-save", default=None,
                   help="save this run's end state as a new named snapshot")
    p.add_argument("--run-id", default=None, help="output directory name under runs/")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--manifest", action="store_true",
                   help="print pinned versions, models and code hash, then exit")
    return p.parse_args()


async def main():
    args = parse_args()
    if args.manifest:
        import json
        from manifest import build_manifest
        print(json.dumps(build_manifest(), indent=2))
        return
    CFG.model = args.model
    CFG.fake_llm = CFG.fake_llm or args.fake_llm
    CFG.ensure_dirs()
    started = time.time()
    # fail before spending anything, not after the run
    if args.state_save and (args.state_save == EMPTY or snapshot_path(args.state_save).exists()):
        raise SnapshotError(f"cannot save state as {args.state_save!r}: reserved or already exists")

    client = LLMClient(CFG)
    specs = build_team(args.agents)
    run = RunContext(client, [s.name for s in specs], state_from=args.state_from)
    adj = topology.build_adjacency(args.topology, len(specs), args.sparsity,
                                   args.seed, args.directed)

    print(f"\nmodel      : {CFG.model}{'  (FAKE)' if CFG.fake_llm else ''}")
    print(f"topology   : {args.topology}  |  schedule: {args.schedule}")
    print(f"state from : {args.state_from}")
    print(f"roster     : {', '.join(f'{i}:{s.name}' for i, s in enumerate(specs))}")
    print(topology.describe(adj))
    print(f"\ntask: {args.task}\n")

    graph = AgentGraph(specs, adj, run)
    await graph.run(args.task, rounds=args.rounds, schedule=args.schedule,
                    verbose=not args.quiet)

    print("\n=== synthesis ===")
    final = await graph.synthesize(args.task)
    print(final)

    run_dir = graph.save(args.run_id, started=started)
    if args.state_save:
        print(f"state saved     : {run.save_snapshot(args.state_save)}")
    print(f"\nblackboard keys : {sorted(graph.blackboard.data)}")
    print(f"files           : {sorted(run.files)}")
    print(f"tool calls      : "
          + ", ".join(f"{a.name}={len(a.tool_log)}" for a in graph.agents))
    print(f"usage           : {run.usage}")
    print(f"run record      : {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
