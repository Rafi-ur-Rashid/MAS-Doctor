# Progress log

One line per finished chunk (EXPERIMENT_PLAN.md §9). Dollars are API spend in that chunk.

| Chunk | Date | Commit | $ spent | Notes |
|---|---|---|---|---|
| C01 | 2026-09-24 | 938a850 | < $0.01 (model probes + 1 tool call) | Open for C02/C03: llm.embed silently falls back to hashed embeddings on API error, which would change retrieval without notice; make it fail loudly in experiment mode. |
| C02 | 2026-09-24 | 59170f6 | $0 | Parallel rounds now run in lockstep (plan §9.3 updated). Live-model path not exercised here; C03's live run covers it. Old files in state/ and old-format runs/*.json are no longer read. |
