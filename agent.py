"""An agent: a persona, a tool subset, its own memory, and a real tool-calling loop."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from config import CFG
from events import Composite, artifact, event, message_artifact, sha256
from memory import AgentMemory
from tools import REGISTRY, ToolCtx


@dataclass
class AgentSpec:
    name: str
    role: str
    instructions: str
    tools: list[str] = field(default_factory=list)

    def system_prompt(self, idx: int, peers: list[str]) -> str:
        peer_line = ", ".join(peers) if peers else "none"
        return (
            f"You are agent_{idx} ({self.name}), the {self.role} in a multi-agent team.\n"
            f"{self.instructions}\n\n"
            f"Other agents in the system: {peer_line}. You only ever see messages from the "
            f"agents connected to you in the communication graph.\n"
            f"Use your tools whenever a fact can be looked up or computed rather than guessed. "
            f"Publish anything the team needs with blackboard_write, and check blackboard_read "
            f"before redoing work someone else may already have done.\n"
            f"Keep replies under 200 words and state your conclusion explicitly."
        )


class Agent:
    def __init__(self, idx: int, spec: AgentSpec, run, peers: list[str],
                 registered_prompt: str | None = None):
        """registered_prompt: the prompt the configuration says this agent has.
        Defaults to the one built from the spec; C10's P-tamper delivers a
        different one than registered."""
        self.idx = idx
        self.spec = spec
        self.run = run
        self.llm = run.llm
        self.name = spec.name
        self.system_prompt = spec.system_prompt(idx, peers)
        registered = registered_prompt if registered_prompt is not None else self.system_prompt
        self.tool_specs = REGISTRY.specs(spec.tools) if spec.tools else []
        self.last_response: str = ""
        self.last_response_id: str | None = None
        self.tool_log: list[dict] = []

        sys_id, msg_id = f"sys:{idx}", f"m:{idx}:0"
        ev = run.events
        ev.emit(artifact(sys_id, "system_prompt", "config", self.system_prompt, agent=self.name))
        ev.emit(message_artifact(msg_id, "runtime", "system", [{"ref": sys_id}], {}))
        ev.emit(event("prompt_delivery", agent=self.name, agent_idx=idx, artifact=sys_id,
                      registered_sha256=sha256(registered),
                      delivered_sha256=sha256(self.system_prompt)))
        self.memory = AgentMemory(self.system_prompt, run.stores[spec.name], msg_id)

    def turn(self, prompt: "Composite | str", round_idx: int = 0) -> "Turn":
        return Turn(self, prompt, round_idx)

    async def act(self, prompt: "Composite | str", round_idx: int = 0) -> str:
        """One whole turn on its own: recall -> LLM -> (tool calls -> LLM)* -> answer -> remember."""
        t = self.turn(prompt, round_idx)
        await t.begin()
        t.flush()
        while not t.done:
            await t.think()
            t.flush()
            await t.execute_tools()
            t.flush()
        text = await t.finish()
        t.flush()
        return text


class Turn:
    """One agent turn, split into steps so a round driver can interleave agents
    deterministically: every agent's LLM call for a step can run concurrently,
    while tool calls, which touch shared state, run one agent at a time in a
    fixed order. The step sequence is exactly the old loop's:

        up to max_tool_iters LLM calls with tools; stop at the first reply
        without tool calls; if every one of them called tools, ask once more
        without tools for a final answer.

    Events go to this turn's buffer; the driver calls flush() after each phase,
    in agent order, so the log's order does not depend on API latency.
    """

    def __init__(self, agent: Agent, prompt: "Composite | str", round_idx: int):
        self.agent = agent
        self.prompt = prompt if isinstance(prompt, Composite) else Composite().text(prompt)
        self.round_idx = round_idx
        self.iters = 0
        self.llm_calls = 0
        self.pending: list = []
        self.pending_call_idx: int | None = None
        self.last_output_id: str | None = None
        self.records: list[dict] = []
        self.text = ""
        self.done = False
        self.buffer: list[dict] = []

    # ------------------------------------------------------------ logging
    def emit(self, record: dict) -> None:
        self.agent.run.events.emit(record, self.buffer)

    def flush(self) -> None:
        self.agent.run.events.flush(self.buffer)

    def _push(self, role: str, parts: list | None, content, fields: dict) -> None:
        """Append a message to working memory and register its recipe."""
        a = self.agent
        aid = f"m:{a.idx}:{len(a.memory.working)}"
        self.emit(message_artifact(aid, a.name, role, parts, fields))
        a.memory.append({"role": role, "content": content, **fields}, aid)

    # ------------------------------------------------------------- phases
    async def begin(self) -> None:
        a = self.agent
        block, read = await a.memory.recall(self.prompt.content)
        self.emit(event("mem_read", agent=a.name, agent_idx=a.idx, round=self.round_idx,
                        query_sha256=sha256(self.prompt.content), **read))
        user = Composite()
        if block is not None:
            user.extend(block).text("\n\n")
        user.extend(self.prompt)
        self._push("user", user.parts, user.content, {})

    async def think(self) -> None:
        """One LLM call. Leaves tool calls in self.pending, or finishes the turn."""
        if self.done:
            return
        a, mem = self.agent, self.agent.memory
        forced = self.iters >= CFG.max_tool_iters
        if forced:
            # tool budget exhausted -- force a text answer
            note = "Tool budget reached. Answer now using what you have."
            self._push("user", [{"text": note}], note, {})
        tool_specs = None if forced else (a.tool_specs or None)
        msgs, ids = mem.window()
        sent_sha = sha256(msgs)
        info: dict = {}
        msg = await a.llm.chat(msgs, tools=tool_specs, info=info)
        self.llm_calls += 1

        calls = None if forced else getattr(msg, "tool_calls", None)
        tool_calls = [{"id": c.id, "type": "function",
                       "function": {"name": c.function.name,
                                    "arguments": c.function.arguments}} for c in calls] if calls else None
        out_id = f"out:{info['idx']}"
        self.emit(artifact(out_id, "llm_output", a.name, msg.content, tool_calls=tool_calls))
        self.emit(event("llm_call", call_idx=info["idx"], agent=a.name, agent_idx=a.idx,
                        round=self.round_idx, step=self.llm_calls, context=ids,
                        tools=[t["function"]["name"] for t in (tool_specs or [])],
                        messages_sha256=sent_sha, cache_key=info.get("cache_key"),
                        cache_hit=info.get("cache_hit"), replica=info.get("replica"),
                        output=out_id))
        self.last_output_id = out_id

        if not forced:
            self.iters += 1
        if not calls:
            self._final(msg.content, out_id)
            return
        parts = None if msg.content is None else [{"ref": out_id}]
        self._push("assistant", parts, msg.content, {"tool_calls": tool_calls})
        self.pending = list(calls)
        self.pending_call_idx = info["idx"]

    async def execute_tools(self) -> None:
        """Run this step's tool calls in the order the model issued them."""
        a = self.agent
        for k, call in enumerate(self.pending):
            ref = f"{self.pending_call_idx}.{k}"
            try:
                args = json.loads(call.function.arguments or "{}")
                parsed = True
            except json.JSONDecodeError:
                args, parsed = {}, False
            self.emit(event("tool_dispatch", agent=a.name, agent_idx=a.idx, round=self.round_idx,
                            call_idx=self.pending_call_idx, k=k, tool_call_id=call.id,
                            tool=call.function.name, arguments=call.function.arguments,
                            args=args, parsed=parsed))
            ctx = ToolCtx(run=a.run, agent_idx=a.idx, agent_name=a.name,
                          round_idx=self.round_idx, ref=ref, emit=self.emit)
            result = await REGISTRY.call(call.function.name, args, ctx)
            tr_id = f"tr:{ref}"
            self.emit(artifact(tr_id, "tool_result", a.name, result,
                               tool=call.function.name, tool_call_id=call.id))
            self.emit(event("tool_result", agent=a.name, agent_idx=a.idx, tool_call_id=call.id,
                            tool=call.function.name, result=tr_id,
                            reads=ctx.reads, writes=ctx.writes))
            self.records.append({"round": self.round_idx, "step": self.iters,
                                 "tool": call.function.name, "args": args,
                                 "result": result[:500]})
            self._push("tool", [{"ref": tr_id}], result,
                       {"tool_call_id": call.id, "name": call.function.name})
        self.pending = []

    async def finish(self) -> str:
        a = self.agent
        resp_id = f"resp:{a.idx}:{self.round_idx}"
        self.emit(artifact(resp_id, "response", a.name, self.text,
                           round=self.round_idx, derived_from=[self.last_output_id]))
        a.last_response = self.text
        a.last_response_id = resp_id
        a.tool_log.extend(self.records)
        if self.text:
            item = await a.memory.remember(self.text, {"kind": "own_answer", "round": self.round_idx})
            if item is not None:
                self.emit(artifact(item.write_id, "memory_item", a.name, item.text,
                                   owner=a.name, meta=item.meta))
                self.emit(event("mem_write", agent=a.name, agent_idx=a.idx, round=self.round_idx,
                                write_id=item.write_id, origin="own_reasoning",
                                visibility="private", derived_from=[resp_id]))
        return self.text

    def _final(self, content, out_id: str) -> None:
        self.text = content or ""
        # "" when the model returned no content: an empty recipe renders to ""
        parts = [{"ref": out_id}] if content is not None else []
        self._push("assistant", parts, self.text, {})
        self.done = True
