# Progress log

One line per finished chunk (EXPERIMENT_PLAN.md §9). Dollars are API spend in that chunk.

| Chunk | Date | Commit | $ spent | Notes |
|---|---|---|---|---|
| C01 | 2026-09-24 | 938a850 | < $0.01 (model probes + 1 tool call) | Open for C02/C03: llm.embed silently falls back to hashed embeddings on API error, which would change retrieval without notice; make it fail loudly in experiment mode. |
| C02 | 2026-09-24 | 59170f6 | $0 | Parallel rounds now run in lockstep (plan §9.3 updated). Live-model path not exercised here; C03's live run covers it. Old files in state/ and old-format runs/*.json are no longer read. |
| C03 | 2026-09-25 | 7453447 | $0.056 upper bound (~$0.039 with cached-input discount, unverified) | Live check: runs/c03_live (59 chat + 25 embed calls) replayed as runs/c03_replay with 0 API calls, transcript sha256 b9b22618... identical. Strict replay of an unrecorded task raised CacheMiss. Default demo run = 59 calls / 132k prompt tokens; confirms the plan's ~$0.04-0.06 per run. PRICES_PER_1M still to be checked against billing. |
| C04 | 2026-09-25 | 99b0b72 | $0 | Strict replay of c03_live with logging: 0 misses, transcript sha256 b9b22618... identical to C03; events.py verify: 59/59 llm_calls rebuilt exactly, 59 cache keys found in store; event log byte-identical across two replays (4eef26f6...). Snapshot format is now 2. |
| C05 | 2026-09-26 | ea9ecff (XG-Guard c05-e0 @ 6603da4) | ~$1.76 (estimated from transcripts; XG-Guard does not meter usage) | G3 PASS: AUC chain 99.56 / tree 99.56 / star 99.11 / random 99.11 vs paper 99.56 / 99.11 / 98.67 / 99.56. Released training file is attacked (98% of listed attackers answer the poison); benign-only training scores 100.00 on all topologies. The paper's AUC = mean per-graph AUROC of binary top-3 flags. See EXPERIMENT_PLAN.md §6 E0 and reports/c05_e0_summary.json. |
