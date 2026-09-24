"""Communication topologies. adj[i][j] == 1 means agent i's messages reach agent j,
so column j gives agent j's in-neighbours -- the same convention XG-Guard uses."""
from __future__ import annotations

import numpy as np

KINDS = ("chain", "ring", "star", "tree", "complete", "random")


def build_adjacency(kind: str, n: int, sparsity: float = 0.5,
                    seed: int | None = None, directed: bool = False) -> np.ndarray:
    adj = np.zeros((n, n), dtype=int)

    if kind == "chain":
        for i in range(n - 1):
            adj[i, i + 1] = 1
            if not directed:
                adj[i + 1, i] = 1

    elif kind == "ring":
        for i in range(n):
            adj[i, (i + 1) % n] = 1
            if not directed:
                adj[(i + 1) % n, i] = 1

    elif kind == "star":                      # agent 0 is the hub
        for i in range(1, n):
            adj[0, i] = 1
            adj[i, 0] = 1

    elif kind == "tree":                      # binary tree over indices
        for i in range(n):
            for child in (2 * i + 1, 2 * i + 2):
                if child < n:
                    adj[i, child] = 1
                    if not directed:
                        adj[child, i] = 1

    elif kind == "complete":
        adj[:, :] = 1
        np.fill_diagonal(adj, 0)

    elif kind == "random":
        rng = np.random.default_rng(seed)
        adj = (rng.random((n, n)) <= sparsity).astype(int)
        np.fill_diagonal(adj, 0)
        if not directed:
            adj = np.maximum(adj, adj.T)
        for i in range(n):                    # no isolated agents
            if adj[i].sum() == 0 and adj[:, i].sum() == 0:
                j = (i + 1) % n
                adj[i, j] = adj[j, i] = 1

    else:
        raise ValueError(f"unknown topology '{kind}', expected one of {KINDS}")

    return adj


def describe(adj: np.ndarray) -> str:
    n = len(adj)
    edges = int(adj.sum())
    lines = [f"{n} agents, {edges} directed edges"]
    for j in range(n):
        srcs = np.nonzero(adj[:, j])[0].tolist()
        lines.append(f"  agent_{j} listens to: {srcs if srcs else '(nobody)'}")
    return "\n".join(lines)
