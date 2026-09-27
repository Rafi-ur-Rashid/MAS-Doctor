"""AgentGraph: round-based message passing over a communication topology."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from agent import Agent, AgentSpec
from cache import KEY_VERSION
from config import CFG
from events import SCHEMA_VERSION, Composite, artifact, event, message_artifact, sha256
from llm import sampling_params


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
    # Prompts are built as Composites: the text the model sees plus its recipe,
    # which records which parts the runtime wrote and which came from artifacts.
    def _initial_prompt(self, task: str) -> Composite:
        return (Composite().text("TASK: ").ref("task", task)
                .text("\n\nThis is round 0. Work the task from your own role using your tools, "
                      "then give your position. Share key findings via blackboard_write."))

    def _round_prompt(self, idx: int, task: str, round_idx: int) -> Composite:
        in_idxs = np.nonzero(self.adj[:, idx])[0]
        digest = Composite()
        if len(in_idxs) == 0:
            digest.text("No other agent is connected to you this round.\n")
        else:
            for src in in_idxs:
                peer = self.agents[src]
                if peer.last_response:
                    if digest.parts:
                        digest.text("\n\n")
                    digest.text(f"--- agent_{src} ({peer.name}) said ---\n")
                    digest.ref(peer.last_response_id, peer.last_response)
                    self.ctx.events.emit(event("msg_recv", round=round_idx, to=self.agents[idx].name,
                                               to_idx=idx, sender=peer.name, sender_idx=int(src),
                                               response=peer.last_response_id))
            if not digest.parts:
                digest.text("Your neighbours produced nothing this round.\n")
        return (Composite().text("TASK (unchanged): ").ref("task", task)
                .text(f"\n\nRound {round_idx}. Messages from the agents connected to you:\n\n")
                .extend(digest)
                .text("\n\nWeigh their input against your own evidence. Verify with tools rather than "
                      "deferring on trust. Then give your updated position, and say plainly if you "
                      "changed your mind and why."))

    def _sent(self, i: int, round_idx: int) -> None:
        """Log agent i's response as sent to its out-neighbours (adj[i][j] == 1)."""
        a = self.agents[i]
        self.ctx.events.emit(event("msg_send", round=round_idx, sender=a.name, sender_idx=i,
                                   response=a.last_response_id,
                                   to=[int(j) for j in np.nonzero(self.adj[i, :])[0]]))

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
        ev = self.ctx.events
        ev.emit(event("run_start", schema=SCHEMA_VERSION, model=CFG.model,
                      sampling=sampling_params(CFG, CFG.model), schedule=schedule,
                      rounds=rounds, adj=self.adj.tolist(),
                      agents=[a.name for a in self.agents], state_from=self.ctx.state_from,
                      toolset=self.ctx.toolset,
                      replica=self.ctx.llm.replica if hasattr(self.ctx.llm, "replica") else 0,
                      replica_from_call=getattr(self.ctx.llm, "replica_from_call", 0)))
        ev.emit(artifact("task", "user_task", "user", task))

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
            out = []
            for i in range(n):
                out.append(await self.agents[i].act(prompt_for(i), round_idx))
                self._sent(i, round_idx)
            return out

        # parallel, in lockstep. Snapshot every prompt first, so nobody reads a
        # peer's fresh reply. After each concurrent phase, event buffers are
        # flushed in agent order, so the log does not depend on API latency.
        turns = [self.agents[i].turn(prompt_for(i), round_idx) for i in range(n)]
        await asyncio.gather(*(t.begin() for t in turns))          # private memory only
        for t in turns:
            t.flush()
        active = turns
        while active:
            await asyncio.gather(*(t.think() for t in active))     # LLM calls, concurrent
            for t in active:
                t.flush()
            for t in active:                                       # shared state, in order
                await t.execute_tools()
                t.flush()
            active = [t for t in active if not t.done]
        # in order: storing a memory ticks the run clock, so its order must not
        # depend on which embedding call returns first
        out = []
        for i, t in enumerate(turns):
            out.append(await t.finish())
            t.flush()
            self._sent(i, round_idx)
        return out

    def _record(self, responses: list[str], verbose: bool) -> None:
        self.communication_data.append([[i, t] for i, t in enumerate(responses)])
        if verbose:
            for i, text in enumerate(responses):
                head = " ".join((text or "").split())[:150]
                print(f"  agent_{i} ({self.agents[i].name}): {head}...")

    MODERATOR_PROMPT = ("You are the moderator of a multi-agent team. Merge the agents' final "
                        "positions into one answer. Note where they agree, resolve disagreements "
                        "by weighing the evidence each cites, and flag anything still unresolved.")

    async def synthesize(self, task: str) -> str:
        ev, bb = self.ctx.events, self.blackboard
        user = Composite().text("TASK: ").ref("task", task).text("\n\nSHARED BLACKBOARD:\n")
        if bb.data:
            for n, (k, v) in enumerate(bb.data.items()):
                if n:
                    user.text("\n")
                # the key is agent-written too, so it is a reference, not runtime text
                user.text("- ").ref(bb.ids[k], k, field="key").text(": ").ref(bb.ids[k], v)
        else:
            user.text("(empty)")
        user.text("\n\nFINAL POSITIONS:\n")
        for n, a in enumerate(self.agents):
            if n:
                user.text("\n\n")
            user.text(f"agent_{a.idx} ({a.name}, {a.spec.role}):\n")
            if a.last_response_id is not None:
                user.ref(a.last_response_id, a.last_response)
        messages = [{"role": "system", "content": self.MODERATOR_PROMPT},
                    {"role": "user", "content": user.content}]
        ids = ["m:mod:0", "m:mod:1"]
        ev.emit(message_artifact(ids[0], "runtime", "system", [{"text": self.MODERATOR_PROMPT}], {}))
        ev.emit(message_artifact(ids[1], "runtime", "user", user.parts, {}))

        info: dict = {}
        msg = await self.llm.chat(messages, info=info)
        out_id = f"out:{info['idx']}"
        ev.emit(artifact(out_id, "llm_output", "moderator", msg.content, tool_calls=None))
        ev.emit(event("llm_call", call_idx=info["idx"], agent="moderator", agent_idx=None,
                      round=None, step=1, context=ids, tools=[],
                      messages_sha256=sha256(messages), cache_key=info.get("cache_key"),
                      cache_hit=info.get("cache_hit"), replica=info.get("replica"),
                      output=out_id))
        self.final_answer = msg.content or ""
        ev.emit(artifact("final", "final_answer", "moderator", self.final_answer,
                         derived_from=[out_id]))
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
        self.ctx.events.write(run_dir / "events.jsonl")
        meta = {"run_id": run_id, "started": started, "finished": time.time(),
                "usage": vars(self.ctx.usage),
                "cache": {**asdict(self.ctx.cache_stats), "path": str(CFG.cache_path),
                          "key_version": KEY_VERSION},
                "budget_usd": CFG.budget_usd,
                "transcript_sha256": hashlib.sha256(
                    (run_dir / "transcript.json").read_bytes()).hexdigest(),
                "events": {"schema": SCHEMA_VERSION, "records": len(self.ctx.events.records),
                           "sha256": hashlib.sha256(
                               (run_dir / "events.jsonl").read_bytes()).hexdigest()},
                "manifest": build_manifest()}
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return run_dir
