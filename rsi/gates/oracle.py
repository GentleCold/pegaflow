"""Offline hit-rate oracle (G3) over an RsiProbe ``requests.jsonl``.

Replays the trace with an unlimited store. A block of request ``j`` ending at
token ``e`` (prompt or generated) is stored at the first scheduler step whose
``progress`` entry has ``computed_after >= e`` and is visible to requests
first scheduled *after* that step (a connector can only save it once the
step's forward ran). Checkpoint groups (Mamba / linear-attention state) only
store a block when a step ended exactly on its boundary: vLLM materializes
state at step ends, never in the middle of a chunk.

The oracle hit of a request is ``semantics.resolve`` over the blocks present
for its prompt at its first schedule.

Group Kinds come from the spec dump, not from the connector under test, so a
connector cannot lower the bar by declaring a stricter Kind:

- attention without a window (Full / MLA / sink): Full
- sliding window ``sw``: Window{sw - 1}; ``sw == 1`` needs no past tokens: Scratch
- Mamba / linear-attention state: Checkpoint
- ``UniformTypeKVCacheSpecs``: the Kind of its inner specs (all must agree)
- anything else: rejected, the gate owner has to map it first.

Per request ``i`` (first episode): ``local_i`` = vLLM's local prefix-cache hit,
``actual_i`` = tokens computed at first schedule (local + external),
``oracle_i`` = oracle hit. The connector is scored on what only it can serve:

- ``ext_i  = max(0, min(actual_i, oracle_i) - local_i)``
- ``ext*_i = max(0, oracle_i - local_i)``
- ``E_ext  = sum(ext) / sum(ext*)``
- ``excess = sum(max(0, actual_i - max(oracle_i, local_i)))``: tokens the
  connector claimed beyond an unlimited store. An honest connector has none
  beyond a partial last block (e.g. reporting ``prompt_len - 1``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pegaflow import semantics as sem

_FULL = {"FullAttentionSpec", "MLAAttentionSpec", "SinkFullAttentionSpec"}
_WINDOW = {"SlidingWindowSpec", "SlidingWindowMLASpec"}
_STATE = {"MambaSpec"}


def _kind(s: dict[str, Any], where: str) -> str:
    """'full' | 'window<N>' | 'scratch' | 'checkpoint' for one spec dict."""
    mro = set(s["mro"])
    if "UniformTypeKVCacheSpecs" in mro:
        inner = {_kind(v, f"{where}/{n}") for n, v in s["kv_cache_specs"].items()}
        if len(inner) != 1:
            raise ValueError(f"oracle: {where} UniformTypeKVCacheSpecs mixes Kinds {inner}")
        return inner.pop()
    sw = s.get("sliding_window")
    if mro & _WINDOW or (mro & _FULL and sw):
        return "scratch" if sw == 1 else f"window{sw - 1}"
    if mro & _FULL:
        return "full"
    if mro & _STATE:
        return "checkpoint"
    raise ValueError(f"oracle has no Kind mapping for {where}: {s['class']}")


def group_kinds(spec: dict[str, Any]) -> list[tuple[str, Any, int]]:
    """(label, semantics.Group | None, block_size) per vLLM group; None = skipped."""
    out = []
    for g in spec["kv_cache_config"]["groups"]:
        s = g["spec"]
        bt = s["block_size"]
        if not g.get("enable_kv_transfer", True) or g.get("is_eagle_group"):
            out.append(("skip", None, bt))
            continue
        label = _kind(s, f"group {g['id']}")
        if label == "full":
            group = sem.Group.full(bt)
        elif label == "scratch":
            group = sem.Group.scratch(bt)
        elif label == "checkpoint":
            group = sem.Group.checkpoint(bt)
        else:
            group = sem.Group.window(bt, int(label.removeprefix("window")))
        out.append((label, group, bt))
    return out


def load_requests(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _store_steps(r: dict[str, Any], bt: int, exact: bool) -> list[tuple[int, int]]:
    """(step, block_end) at which each complete block of ``r`` was first computed.

    ``progress`` is in step order; after a preemption it restarts lower, so
    the earliest entry reaching the block end wins. ``exact`` (checkpoint):
    only an entry ending exactly on the block end counts.
    """
    out = []
    for e in range(bt, r["computed"] + 1, bt):
        hit = (p[0] for p in r["progress"] if (p[1] == e if exact else p[1] >= e))
        step = next(hit, None)
        if step is not None:
            out.append((step, e))
    return out


def replay(spec: dict[str, Any], reqs: list[dict[str, Any]]) -> list[int]:
    """Oracle resumable tokens for each request (first episode), in input order."""
    kinds = [(label, g, bt) for label, g, bt in group_kinds(spec) if g is not None]
    if not kinds or all(label == "scratch" for label, _, _ in kinds):
        return [0] * len(reqs)

    # Store events: (visible_after_step, group, key).
    events = []
    for r in reqs:
        kt = r["key_tokens"]
        for gi, (label, _, bt) in enumerate(kinds):
            if label == "scratch":
                continue
            for step, e in _store_steps(r, bt, exact=label == "checkpoint"):
                events.append((step, gi, r["keys"][e // kt - 1]))
    events.sort()
    order = sorted(range(len(reqs)), key=lambda i: (reqs[i]["first_step"] or 0, i))
    stored: list[set[str]] = [set() for _ in kinds]
    ei = 0
    result = [0] * len(reqs)
    for i in order:
        r = reqs[i]
        first = r["first_step"] or 0
        while ei < len(events) and events[ei][0] < first:
            _, gi, key = events[ei]
            stored[gi].add(key)
            ei += 1
        kt = r["key_tokens"]
        present = []
        for gi, (_, _, bt) in enumerate(kinds):
            n = min(r["prompt_len"], r["computed"]) // bt
            present.append([b for b in range(n) if r["keys"][(b + 1) * bt // kt - 1] in stored[gi]])
        result[i] = sem.resolve([g for _, g, _ in kinds], present, r["prompt_len"])
    return result


def summarize(
    spec: dict[str, Any],
    reqs: list[dict[str, Any]],
    segments: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Hit numbers and ``E_ext`` per segment (name -> set of ``rsi_pass`` labels).

    The oracle always replays the whole trace; segments only select which
    requests are scored. ``None`` scores everything as ``all``.
    """
    oracle = replay(spec, reqs)
    kinds = group_kinds(spec)
    block = max((bt for _, g, bt in kinds if g is not None), default=0)
    out: dict[str, Any] = {"kinds": [label for label, _, _ in kinds]}
    for name, labels in (segments or {"all": None}).items():
        part = [i for i, r in enumerate(reqs) if labels is None or r.get("pass") in labels]
        prompt = actual = local = best = ext = ext_star = excess = 0
        for i in part:
            r = reqs[i]
            a, lo, o = r["first_computed"] or 0, r["local_hit"] or 0, oracle[i]
            prompt += r["prompt_len"]
            actual += a
            local += lo
            best += o
            ext += max(0, min(a, o) - lo)
            ext_star += max(0, o - lo)
            excess += max(0, a - max(o, lo))
        out[name] = {
            "requests": len(part),
            "prompt_tokens": prompt,
            "hit_tokens": actual,
            "local_hit_tokens": local,
            "external_hit_tokens": actual - local,
            "external_hit_rate": (actual - local) / (prompt - local) if prompt > local else 0.0,
            "oracle_tokens": best,
            "oracle_hit_rate": best / prompt if prompt else 0.0,
            "hit_rate": actual / prompt if prompt else 0.0,
            "ext_tokens": ext,
            "ext_oracle_tokens": ext_star,
            "efficiency": ext / ext_star if ext_star else None,
            "excess_tokens": excess,
            "excess_limit": block * len(part),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("probe_dir", type=Path)
    ap.add_argument("--by-pass", action="store_true", help="split by the rsi_pass tag")
    args = ap.parse_args(argv)
    spec = json.loads((args.probe_dir / "spec.json").read_text())
    reqs = load_requests(args.probe_dir / "requests.jsonl")
    passes = {r.get("pass") or "" for r in reqs}
    segments = {p: {p} for p in sorted(passes)} if args.by_pass else None
    print(json.dumps(summarize(spec, reqs, segments), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
