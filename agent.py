"""An agent: a persona, a tool subset, its own memory, and a real tool-calling loop."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from config import CFG
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
    def __init__(self, idx: int, spec: AgentSpec, run, peers: list[str]):
        self.idx = idx
        self.spec = spec
        self.run = run
        self.llm = run.llm
        self.name = spec.name
        self.system_prompt = spec.system_prompt(idx, peers)
        self.memory = AgentMemory(self.system_prompt, run.stores[spec.name])
        self.tool_specs = REGISTRY.specs(spec.tools) if spec.tools else []
        self.last_response: str = ""
        self.tool_log: list[dict] = []

    def turn(self, prompt: str, round_idx: int = 0) -> "Turn":
        return Turn(self, prompt, round_idx)

    async def act(self, prompt: str, round_idx: int = 0) -> str:
        """One whole turn on its own: recall -> LLM -> (tool calls -> LLM)* -> answer -> remember."""
        t = self.turn(prompt, round_idx)
        await t.begin()
        while not t.done:
            await t.think()
            await t.execute_tools()
        return await t.finish()


class Turn:
    """One agent turn, split into steps so a round driver can interleave agents
    deterministically: every agent's LLM call for a step can run concurrently,
    while tool calls, which touch shared state, run one agent at a time in a
    fixed order. The step sequence is exactly the old loop's:

        up to max_tool_iters LLM calls with tools; stop at the first reply
        without tool calls; if every one of them called tools, ask once more
        without tools for a final answer.
    """

    def __init__(self, agent: Agent, prompt: str, round_idx: int):
        self.agent = agent
        self.prompt = prompt
        self.round_idx = round_idx
        self.iters = 0
        self.pending: list = []
        self.records: list[dict] = []
        self.text = ""
        self.done = False

    async def begin(self) -> None:
        mem = self.agent.memory
        recalled = await mem.recall(self.prompt)
        content = f"{recalled}\n\n{self.prompt}" if recalled else self.prompt
        mem.append({"role": "user", "content": content})

    async def think(self) -> None:
        """One LLM call. Leaves tool calls in self.pending, or finishes the turn."""
        if self.done:
            return
        a, mem = self.agent, self.agent.memory
        if self.iters >= CFG.max_tool_iters:
            # tool budget exhausted -- force a text answer
            mem.append({"role": "user",
                        "content": "Tool budget reached. Answer now using what you have."})
            msg = await a.llm.chat(mem.prompt_messages(), tools=None)
            self._final(msg.content or "")
            return

        msg = await a.llm.chat(mem.prompt_messages(), tools=a.tool_specs or None)
        self.iters += 1
        calls = getattr(msg, "tool_calls", None)
        if not calls:
            self._final(msg.content or "")
            return
        mem.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name,
                                         "arguments": c.function.arguments}} for c in calls],
        })
        self.pending = list(calls)

    async def execute_tools(self) -> None:
        """Run this step's tool calls in the order the model issued them."""
        a = self.agent
        ctx = ToolCtx(run=a.run, agent_idx=a.idx, agent_name=a.name, round_idx=self.round_idx)
        for call in self.pending:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = await REGISTRY.call(call.function.name, args, ctx)
            self.records.append({"round": self.round_idx, "step": self.iters,
                                 "tool": call.function.name, "args": args,
                                 "result": result[:500]})
            a.memory.append({"role": "tool", "tool_call_id": call.id,
                             "name": call.function.name, "content": result})
        self.pending = []

    async def finish(self) -> str:
        a = self.agent
        a.last_response = self.text
        a.tool_log.extend(self.records)
        if self.text:
            await a.memory.remember(self.text, {"kind": "own_answer", "round": self.round_idx})
        return self.text

    def _final(self, text: str) -> None:
        self.text = text
        self.agent.memory.append({"role": "assistant", "content": text})
        self.done = True
