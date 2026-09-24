"""An agent: a persona, a tool subset, its own memory, and a real tool-calling loop."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from config import CFG
from memory import AgentMemory, CURRENT_AUTHOR
from tools import REGISTRY


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
    def __init__(self, idx: int, spec: AgentSpec, llm, peers: list[str], persist: bool = True):
        self.idx = idx
        self.spec = spec
        self.llm = llm
        self.name = spec.name
        self.system_prompt = spec.system_prompt(idx, peers)
        self.memory = AgentMemory(idx, self.system_prompt, llm, persist=persist)
        self.tool_specs = REGISTRY.specs(spec.tools) if spec.tools else []
        self.last_response: str = ""
        self.tool_log: list[dict] = []

    async def act(self, prompt: str, round_idx: int = 0) -> str:
        """One turn: recall -> LLM -> (tool calls -> LLM)* -> answer -> remember."""
        CURRENT_AUTHOR.set(self.name)

        recalled = await self.memory.recall(prompt)
        content = f"{recalled}\n\n{prompt}" if recalled else prompt
        self.memory.append({"role": "user", "content": content})

        turn_tools: list[dict] = []
        text = ""
        for _ in range(CFG.max_tool_iters):
            msg = await self.llm.chat(self.memory.prompt_messages(),
                                      tools=self.tool_specs or None)
            calls = getattr(msg, "tool_calls", None)

            if not calls:
                text = msg.content or ""
                self.memory.append({"role": "assistant", "content": text})
                break

            self.memory.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name,
                                             "arguments": c.function.arguments}} for c in calls],
            })
            for call in calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = await REGISTRY.call(call.function.name, args)
                turn_tools.append({"tool": call.function.name, "args": args,
                                   "result": result[:500]})
                self.memory.append({"role": "tool", "tool_call_id": call.id,
                                    "name": call.function.name, "content": result})
        else:
            # tool budget exhausted -- force a text answer
            self.memory.append({"role": "user",
                                "content": "Tool budget reached. Answer now using what you have."})
            msg = await self.llm.chat(self.memory.prompt_messages(), tools=None)
            text = msg.content or ""
            self.memory.append({"role": "assistant", "content": text})

        self.last_response = text
        self.tool_log.extend(turn_tools)
        if text:
            await self.memory.remember(text, {"kind": "own_answer", "round": round_idx})
        return text
