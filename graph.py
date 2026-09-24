"""AgentGraph: round-based message passing over a communication topology."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np

from agent import Agent, AgentSpec
from config import CFG
from memory import Blackboard


class AgentGraph:
    def __init__(self, specs: list[AgentSpec], adj: np.ndarray, llm, persist: bool = True):
        assert len(specs) == len(adj), "one spec per node required"
        self.adj = adj
        self.llm = llm
        names = [s.name for s in specs]
        self.agents = [
            Agent(i, s, llm, peers=[n for k, n in enumerate(names) if k != i], persist=persist)
            for i, s in enumerate(specs)
        ]
        self.blackboard = Blackboard(persist=persist)
        self.communication_data: list[list[list]] = []

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
        """schedule='parallel'   -- every agent acts at once on a snapshot of the
        previous round (fast; what XG-Guard does). Agents cannot see work their
        peers do in the same round, so early blackboard reads can come up empty.

        schedule='sequential' -- agents act in index order within a round, each
        seeing whatever its neighbours already produced this round. Slower, but
        downstream roles (a writer, a reporter) actually have something to consume.
        """
        self.communication_data = []

        if verbose:
            print(f"\n=== round 0: independent work "
                  f"({len(self.agents)} agents, {schedule}) ===")
        responses = await self._step([self._initial_prompt(task)] * len(self.agents),
                                     0, schedule, task, initial=True)
        self._record(responses, verbose)

        for r in range(1, rounds + 1):
            if verbose:
                print(f"\n=== round {r}: neighbour exchange ({schedule}) ===")
            responses = await self._step(None, r, schedule, task)
            self._record(responses, verbose)

        return self.communication_data

    async def _step(self, prompts, round_idx, schedule, task, initial=False):
        n = len(self.agents)

        def prompt_for(i):
            if prompts is not None:
                return prompts[i]
            return self._round_prompt(i, task, round_idx)

        if schedule == "parallel":
            # snapshot every prompt first, so nobody reads a peer's fresh reply
            fixed = [prompt_for(i) for i in range(n)]
            return await asyncio.gather(
                *(self.agents[i].act(fixed[i], round_idx) for i in range(n)))

        if schedule == "sequential":
            out = []
            for i in range(n):
                out.append(await self.agents[i].act(prompt_for(i), round_idx))
            return out

        raise ValueError(f"unknown schedule '{schedule}' (parallel|sequential)")

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
        return msg.content or ""

    # ------------------------------------------------------------ transcript
    def save(self, task: str, final_answer: str, out_dir: Path | None = None) -> Path:
        out_dir = out_dir or CFG.runs_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "task": task,
            "model": CFG.model,
            "adj_matrix": self.adj.tolist(),
            "system_prompts": [a.system_prompt for a in self.agents],
            "agent_names": [a.name for a in self.agents],
            "agent_roles": [a.spec.role for a in self.agents],
            "communication_data": self.communication_data,
            "tool_calls": {a.name: a.tool_log for a in self.agents},
            "blackboard": self.blackboard.data,
            "blackboard_log": self.blackboard.log,
            "final_answer": final_answer,
            "usage": str(self.llm.usage),
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path
