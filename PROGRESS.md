# Progress log

One line per finished chunk (EXPERIMENT_PLAN.md §9). Dollars are API spend in that chunk.

| Chunk | Date | Commit | $ spent | Notes |
|---|---|---|---|---|
| C01 | 2026-09-24 | 938a850 | < $0.01 (model probes + 1 tool call) | Open for C02/C03: llm.embed silently falls back to hashed embeddings on API error, which would change retrieval without notice; make it fail loudly in experiment mode. |
| C02 | 2026-09-24 | 59170f6 | $0 | Parallel rounds now run in lockstep (plan §9.3 updated). Live-model path not exercised here; C03's live run covers it. Old files in state/ and old-format runs/*.json are no longer read. |
| C03 | 2026-09-25 | 7453447 | $0.056 upper bound (~$0.039 with cached-input discount, unverified) | Live check: runs/c03_live (59 chat + 25 embed calls) replayed as runs/c03_replay with 0 API calls, transcript sha256 b9b22618... identical. Strict replay of an unrecorded task raised CacheMiss. Default demo run = 59 calls / 132k prompt tokens; confirms the plan's ~$0.04-0.06 per run. PRICES_PER_1M still to be checked against billing. |
