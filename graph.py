"""AgentGraph: round-based message passing over a communication topology."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np

from agent import Agent, AgentSpec
from config import CFG


class AgentGraph:
    def __init__(self, specs: list[AgentSpec], adj: np.ndarray, ctx):
        """ctx: this run's RunContext. All mutable state lives there."""
        assert len(specs) == len(adj), "one spec per node required"
        self.adj = adj
        self.ctx = ctx
        self.llm = ctx.llm
        self.blackboard = ctx.blackboard
        names = [s.name for s in specs]
        assert len(set(names)) == len(names), "agent names must be unique"
        self.agents = [
            Agent(i, s, ctx, peers=[n for k, n in enumerate(names) if k != i])
            for i, s in enumerate(specs)
        ]
        self.communication_data: list[list[list]] = []
        self.final_answer: str | None = None
        self.task: str | None = None
        self.schedule: str | None = None

    # ------------------------------------------------------------ prompting
    def _initial_prompt(self, task: str) -> str:
        return (f"TASK: {task}\n\n"
                f"This is round 0. Work the task from your own role using your tools, "
                f"then give your position. Share key findings via blackboard_write.")

    def _round_prompt(self, idx: int, task: str, round_idx: int) -> str:
        in_idxs = np.nonzero(self.adj[:, idx])[0]
        if len(in_idxs) == 0:
            digest = "No other agent is connected to you this round.\n"
        else:
            parts = []
            for src in in_idxs:
                peer = self.agents[src]
                if peer.last_response:
                    parts.append(f"--- agent_{src} ({peer.name}) said ---\n{peer.last_response}")
            digest = "\n\n".join(parts) or "Your neighbours produced nothing this round.\n"
        return (f"TASK (unchanged): {task}\n\n"
                f"Round {round_idx}. Messages from the agents connected to you:\n\n{digest}\n\n"
                f"Weigh their input against your own evidence. Verify with tools rather than "
                f"deferring on trust. Then give your updated position, and say plainly if you "
                f"changed your mind and why.")

    # ----------------------------------------------------------------- runs
    async def run(self, task: str, rounds: int = 2, schedule: str = "parallel",
                  verbose: bool = True) -> list[list[list]]:
        """schedule='parallel'   -- every agent acts on a snapshot of the previous
        round's messages (what XG-Guard does). Within the round, agents proceed in
        lockstep: all of them make their next LLM call concurrently, then their tool
        calls run one agent at a time in index order. Shared state therefore changes
        in the same order however fast each API call returns, so a run is repeatable.
        Tool effects are visible to other agents from the next step on.

        schedule='sequential' -- agents act in index order within a round, each
        seeing whatever its neighbours already produced this round. Slower, but
        downstream roles (a writer, a reporter) actually have something to consume.
        """
        if schedule not in ("parallel", "sequential"):
            raise ValueError(f"unknown schedule '{schedule}' (parallel|sequential)")
        self.task, self.schedule = task, schedule
        self.communication_data = []

        if verbose:
            print(f"\n=== round 0: independent work "
                  f"({len(self.agents)} agents, {schedule}) ===")
        responses = await self._step([self._initial_prompt(task)] * len(self.agents),
                                     0, schedule, task)
        self._record(responses, verbose)

        for r in range(1, rounds + 1):
            if verbose:
                print(f"\n=== round {r}: neighbour exchange ({schedule}) ===")
            responses = await self._step(None, r, schedule, task)
            self._record(responses, verbose)

        return self.communication_data

    async def _step(self, prompts, round_idx, schedule, task):
        n = len(self.agents)

        def prompt_for(i):
            if prompts is not None:
                return prompts[i]
            return self._round_prompt(i, task, round_idx)

        if schedule == "sequential":
            return [await self.agents[i].act(prompt_for(i), round_idx) for i in range(n)]

        # parallel, in lockstep. Snapshot every prompt first, so nobody reads a
        # peer's fresh reply.
        turns = [self.agents[i].turn(prompt_for(i), round_idx) for i in range(n)]
        await asyncio.gather(*(t.begin() for t in turns))          # private memory only
        active = turns
        while active:
            await asyncio.gather(*(t.think() for t in active))     # LLM calls, concurrent
            for t in active:                                       # shared state, in order
                await t.execute_tools()
            active = [t for t in active if not t.done]
        # in order: storing a memory ticks the run clock, so its order must not
        # depend on which embedding call returns first
        return [await t.finish() for t in turns]

    def _record(self, responses: list[str], verbose: bool) -> None:
        self.communication_data.append([[i, t] for i, t in enumerate(responses)])
        if verbose:
            for i, text in enumerate(responses):
                head = " ".join((text or "").split())[:150]
                print(f"  agent_{i} ({self.agents[i].name}): {head}...")

    async def synthesize(self, task: str) -> str:
        board = self.blackboard.data
        positions = "\n\n".join(
            f"agent_{a.idx} ({a.name}, {a.spec.role}):\n{a.last_response}" for a in self.agents)
        shared = "\n".join(f"- {k}: {v}" for k, v in board.items()) or "(empty)"
        messages = [
            {"role": "system",
             "content": "You are the moderator of a multi-agent team. Merge the agents' final "
                        "positions into one answer. Note where they agree, resolve disagreements "
                        "by weighing the evidence each cites, and flag anything still unresolved."},
            {"role": "user",
             "content": f"TASK: {task}\n\nSHARED BLACKBOARD:\n{shared}\n\nFINAL POSITIONS:\n{positions}"},
        ]
        msg = await self.llm.chat(messages)
        self.final_answer = msg.content or ""
        return self.final_answer

    # ------------------------------------------------------------ transcript
    def transcript(self) -> dict:
        """Everything the run produced. Contains no wall-clock time and no run id,
        so the same run from the same state yields a byte-identical file."""
        return {
            "task": self.task,
            "model": CFG.model,
            "schedule": self.schedule,
            "state_from": self.ctx.state_from,
            "adj_matrix": self.adj.tolist(),
            "system_prompts": [a.system_prompt for a in self.agents],
            "agent_names": [a.name for a in self.agents],
            "agent_roles": [a.spec.role for a in self.agents],
            "communication_data": self.communication_data,
            "tool_calls": {a.name: a.tool_log for a in self.agents},
            "blackboard": self.blackboard.data,
            "blackboard_log": self.blackboard.log,
            "files": self.ctx.files,
            "final_answer": self.final_answer,
        }

    def save(self, run_id: str | None = None, out_dir: Path | None = None,
             started: float | None = None) -> Path:
        """Writes runs/<run_id>/transcript.json (deterministic) and meta.json
        (wall-clock times, usage, manifest). Refuses to overwrite a run."""
        from manifest import build_manifest   # imported here: manifest imports llm/config only
        run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        run_dir = (out_dir or CFG.runs_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "transcript.json").write_text(json.dumps(self.transcript(), indent=2))
        meta = {"run_id": run_id, "started": started, "finished": time.time(),
                "usage": vars(self.ctx.usage), "manifest": build_manifest()}
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return run_dir
