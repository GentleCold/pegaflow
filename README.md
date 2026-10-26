# rsi-notes: RSI working memory

This orphan branch holds the agents' working memory for the RSI effort
(vLLM connector_v2 + PegaFlow, model onboarding in RSI mode):

- `PROGRESS.md`: progress log, scope, phases, open questions, server
  limitations, and the dated user decisions behind the gate rules.
- `models/<model>/ITERATIONS.md`: per-model iteration journal (hypotheses,
  run tags, hit-rate judgments, review outcomes).
- `reviews/*.md`: review reports that still have unresolved findings.
- `sha-map-2026-10-24.md`: old -> new commit ids of the 2026-10-24
  commit-message rewrite (run records keep the old ids).

It is never merged into a code branch. Code branches (`feat/semantics-rsi`,
`feat/rsi-<model>`, gate branches) carry only current-state docs
(`docs/rsi/GATES.md`, `docs/rsi/models/<model>/{spec.json,gate.json}`, the
`pegaflow-new-model` skill, package READMEs): no dated history, change logs
or journals.

## Use

Worktree: `.worktrees/rsi-notes` (`git -C .worktrees/rsi-notes ...`).
Several agents share this branch: commit small, `git pull --rebase` before
every push, never force-push. Commit messages follow commitizen
(`docs(rsi): ...`).

## Compaction rules

- Budgets: `PROGRESS.md` <= 300 lines, each `ITERATIONS.md` <= 400 lines.
- When a file is over budget, and at every model acceptance, rewrite it
  instead of appending:
  - current state first;
  - each closed iteration compacted to one table row (iteration,
    hypothesis, result, run tag, commit);
  - dead ends kept as one line each, so they are not retried;
  - superseded numbers dropped.
  The branch's git history keeps the full text.
- Review reports: a review file lives only until its findings are resolved.
  Then it is deleted and replaced by one line in the model's `ITERATIONS.md`
  (verdict, key findings, fix commits). A reviewer may also just return its
  report to the orchestrator without writing a file.
- Not yet applied to the imported files: the model branches
  `feat/rsi-deepseek-v2-lite` and `feat/rsi-qwen3.5-4b` still carry
  unmerged journal edits; they are migrated here when those branches fold
  in, and the compaction happens after that.
