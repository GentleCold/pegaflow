# G5 gate re-review 4

Scope: `d6a5da0..8d529fd -- rsi/gates` (HEAD 8d529fd), plus the GATES.md
diff. Reviewed from the code diff, not commit messages. Line numbers are
at 8d529fd.

## Item table

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | M4-M7, S7-S9, N5 closure | closed, M7 partly | all code present at HEAD; M7 independent-signal claim has a hole (M9) |
| 2 | Official-run integrity | open | `rsi.*` connector for the arm under test escapes the snapshot check (M8) |
| 3 | Gate rules not relaxed | pass | only tightenings; `modules` moved from G0.spec to G0.provenance |
| 4 | Isolation residual | pending | |
| 5 | Teardown / ports | pending | |
| 6 | Evidence | pending | |
| 7 | Tests | pending | |

## 1. Closure of re-review 3 items

Code for every item is in the HEAD tree. The S8/S9 code lost in d90ac5d /
1b640f7 is present (93a654f, 47cbd44).

| item | status | evidence |
|---|---|---|
| M4 | closed | `run_gate.py:1027` `_VOLATILE` has `modules`; test `test_g0_spec_ignores_module_paths` (`test_oracle_capacity.py:1416`); GPU3 run G0.spec true (item 6) |
| M5 | closed | `provenance.py:116-134` walks every dir, lists symlinks, never follows; `check` (`:152-194`) rejects symlinks, untracked, other `.so`, `.pyc` unless `pycache_ok`; `stage` (`:269-315`) verifies the source with `pycache_ok=False` (`:282`), copies only rev blobs + the native module, re-verifies the copy and compares native sha256 (`:311`). Stage processes get `PYTHONPYCACHEPREFIX` (`run_gate.py:446`, `:520`, `:661`); the gate re-execs with one (`:2312-2321`). The native module itself is still not tied to a rev (N7) |
| M6 | closed | `provenance_problems` (`run_gate.py:2162-2176`) adds gate and venv trees, missing record included; wired at `:2423` |
| M7a | closed as deterrence | `isolate.py` (mount ns, tmpfs over runs root, rw arm dir, ro snapshot, cap drop); `isolated()` for both server and vLLM of the arm under test (`run_gate.py:542`, `:739`); non-isolated arm is non-official (`:2436`) |
| M7b | closed | `provenance.scan` + `SCAN_ALLOWLIST` (empty) (`provenance.py:324-394`); flagged hits non-official (`run_gate.py:2426`) |
| M7c | closed, with overclaims | GATES.md "Boundary (M7)"; skill checklist `.claude/skills/pegaflow-new-model/SKILL.md:87-89`. See M9, M10 |
| M7d | closed | `VLLM_PLUGINS=""` for every vLLM (`run_gate.py:671`) |
| S7 | closed | `aligned` treats a missing record as not aligned (`run_gate.py:1201`); `digest_compare.compare` counts `unjudged_nonaligned`, a missing `prompt_len` counts (`digest_compare.py:203-205`); warm requires 0 (`run_gate.py:1314`); `_hit_alignment` defaults to 16, so the count is always on |
| S8 | closed | baseline p99 vs the arms' median (`run_gate.py:1956-1961`) |
| S9 | closed | `G4.warmup_seconds` per workload, no `pass` key, so not gated (`run_gate.py:1575-1582`); PROGRESS open question at `docs/rsi/PROGRESS.md:584` |
| N5 | closed | Popen-to-`_LIVE` window deferred (`run_gate.py:342-353`), honored by `check_stop()` inside the `try`; residual windows leave the group in `_LIVE`, stopped by `stop_live` (`:300-306`, atexit `:2090`); `stop` tolerates `fh=None` (`AttributeError` suppressed) |

## 2. Official-run integrity

| condition | can `official: true` happen? | evidence |
|---|---|---|
| under-test tree unverified | no | `stage` returns `verified: False` on any source or copy mismatch; a missing package raises (no summary); `provenance_problems` |
| gate / venv trees unverified | no (at start); yes if changed mid-run | checked once before stage 1 (`run_gate.py:2333`), never re-checked. See M9 |
| connector `__file__` outside the snapshot | yes, for an `rsi.*` module | M8 |
| CUDA env mismatch | no for the gate's env | `cuda_problems` (`:2043-2060`), wired at `:2424`; unknown torch CUDA is a problem. Stage PATH differs from the checked PATH (N8) |
| `replay_skip > 0` | no | `OFFICIAL["replay_skip"] = 0` (`:72`), equality check in `official_problems` |
| a PortRace | yes, in a reference arm | a required arm re-raises (`:2210`), so no summary; a record-only arm's PortRace is recorded and the run stays official (S11) |

No swallowed exception turns a failed check into a pass: `Trees._fetched`
returns a reason string (unverified), `listing` returns `None` (unverified),
`record_provenance` has no `except`. `cuda_env` suppresses the torch
subprocess error, which leaves `torch_cuda=None`, which is a problem.

## 3. Gate rules not relaxed

- `OFFICIAL` at d6a5da0 vs 8d529fd: one key added, `replay_skip: 0`. No value
  changed.
- No `MIN_*` / `MAX_*` / `CONTAMINATION_P99` / `CAPACITY_*` constant changed in
  the diff.
- GATES.md: every removed line with a number is reworded with the same
  number (`> 10x`, coverage `0.9`). Added rules only tighten: two-sided
  contamination, zero unjudged non-aligned warm blocks, provenance of gate
  and venv trees, connector `__file__`, `--under-test-path` required,
  isolation required, CUDA preflight, scan.
- `_VOLATILE += modules` drops a G0.spec comparison. G0.provenance judges the
  same paths (vLLM / torch / prefix under the venv, probe under the gate
  tree, `pegaflow` under the expected dir). Moved, not relaxed.

## Must-fix

### M8. The arm under test can name an `rsi.*` module and skip the snapshot check

`g0_provenance` picks the connector's root by module name, not by arm kind
(`run_gate.py:1128`):

```python
root = REPO.resolve() if connector.startswith("rsi.") else pegaflow_dir
```

`arm_kind` returns `under_test` for any `<module>:<Class>` that does not start
with `selftest:` (`:581-584`). Scenario: an agent commits a connector under
`rsi/` on its branch, pushes it (so the gate tree verifies against origin),
and runs `--arms plain,baseline,rsi.gates.myconn:MyConn --under-test-path
<clean tree>`. The snapshot is clean and scanned. The connector that actually
runs lives in the gate tree, is never scanned (`scan_record` scans only the
snapshot), and G0.provenance accepts it because its file is under `REPO`. The
run can be official while the measured connector is not the one in the
snapshot.

Fix: choose the root by kind. For `under_test`, the connector must be under
the snapshot. Only `selftest` maps to the gate tree. Also make
`official_problems` reject an under-test module that does not resolve to
the snapshot's `pegaflow` package. The test at `test_oracle_capacity.py:1191`
already says "the gate tree for `rsi.*` self-tests"; add a case for an
under-test arm named `rsi.x:Y`.
