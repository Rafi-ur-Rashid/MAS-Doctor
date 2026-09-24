"""Central configuration for the multi-agent system."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # --- LLM ---
    # Dated snapshots, not aliases: an alias like "gpt-5-mini" can be repointed
    # mid-experiment. See EXPERIMENT_PLAN.md §2.1.
    model: str = os.getenv("MAS_MODEL", "gpt-5-mini-2025-08-07")
    xgguard_model: str = "gpt-4o-mini-2024-07-18"   # E0 only: the model XG-Guard's data was made with
    embed_model: str = os.getenv("MAS_EMBED_MODEL", "text-embedding-3-small")
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or None
    )
    # Non-reasoning models (gpt-4o family) take these two ...
    temperature: float = 0.2
    max_tokens: int = 1024
    # ... reasoning models (gpt-5 family) reject both and take these instead.
    # max_completion_tokens also covers hidden reasoning tokens, hence the larger budget.
    reasoning_effort: str = os.getenv("MAS_REASONING_EFFORT", "low")
    max_completion_tokens: int = 4096
    # If the embedding API fails: False = raise (experiments), True = hashed fallback (demos).
    embed_fallback: bool = bool(os.getenv("MAS_EMBED_FALLBACK"))

    # --- benchmark ---
    agentdojo_benchmark: str = "v1.2.2"
    agentdojo_suite: str = "workspace"
    xgguard_dir: Path = Path(os.getenv("XGGUARD_DIR", "/scratch/mur5028/XG-Guard"))

    # --- orchestration ---
    max_tool_iters: int = 5      # tool-calling loop depth per agent turn
    max_concurrency: int = 6     # parallel in-flight LLM requests
    max_retries: int = 4
    working_memory_turns: int = 12   # user/assistant pairs kept in the prompt
    recall_k: int = 3                # max episodic memories injected per turn
    recall_min_similarity: float = 0.25   # below this, a memory is not relevant

    # --- paths ---
    root: Path = ROOT
    kb_dir: Path = ROOT / "knowledge_base"    # corpus for kb_search
    state_dir: Path = Path(os.getenv("MAS_STATE_DIR", ROOT / "state"))   # named state snapshots
    runs_dir: Path = Path(os.getenv("MAS_RUNS_DIR", ROOT / "runs"))   # transcripts

    fake_llm: bool = bool(os.getenv("MAS_FAKE_LLM"))

    def ensure_dirs(self) -> None:
        for d in (self.kb_dir, self.state_dir, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)


CFG = Config()
