---
name: debug
description: >-
  Diagnose and safely repair recoverable failures during project execution
  (paths, metadata, graphs, env, SLURM config) without altering scientific
  conclusions; escalate with debug-report.md when unsafe or impossible.
  Invoked by @verify-todo (Recoverable graph), @do (after monitor escalation /
  RECOVERABLE tasks), or @prompt-orchestrator (within-task Safe recovery).
  Not invoked by @monitor.
disable-model-invocation: true
---

# Debug

## Purpose

Automatically diagnose and repair recoverable failures detected during project execution.

**Callers (must invoke explicitly):**

| Caller | When |
|--------|------|
| `@verify-todo` | Recoverable graph / metadata issues |
| `@do` | After `@monitor` escalates a job failure, or task Status is **RECOVERABLE**, before returning to 4.1 |
| `@prompt-orchestrator` | Within-task Phase 6 when Safe infrastructure recovery is needed |

**Not a caller:** `@monitor` (status-only; escalates to `@do` instead).

It should attempt safe automatic recovery without modifying scientific conclusions.

Follow project rules: **scientific-integrity**, **validation-first**, **missing-data-policy**, **method-decision-tracking**, **reproducibility**, **task-status**. Never invent missing data or fabricate repair success.

## Checklist

```
Debug:
- [ ] Inputs inspected
- [ ] Phase 1: Diagnose
- [ ] Phase 2: Recovery planning (Safe / Unsafe / Impossible)
- [ ] Phase 3: Repair (Safe only)
- [ ] Phase 4: Validation
- [ ] Phase 5: Escalation if failed → debug-report.md
```

==================================================
Inputs
==================================================

Inspect:

- error messages
- logs
- reports
- dependency graph
- project structure
- task specifications
- method-decision.md
- artifact-registry.md

==================================================
Phase 1 — Diagnose
==================================================

Determine the root cause.

Possible causes include:

- missing files
- missing dependencies
- incorrect paths
- invalid task metadata
- dependency graph inconsistencies
- software configuration
- environment activation
- SLURM submission
- resource allocation
- workflow configuration

Record: observed symptoms → hypothesized root cause → evidence (log lines, paths, report excerpts). Do not invent causes without evidence.

==================================================
Phase 2 — Recovery Planning
==================================================

Determine whether recovery is:

- Safe
- Unsafe
- Impossible

Only Safe recovery may proceed automatically.

| Classification | Meaning | Action |
|----------------|---------|--------|
| **Safe** | Minimal infrastructure/metadata repair; no change to scientific results or conclusions; reversible or clearly scoped | Proceed to Phase 3 |
| **Unsafe** | Would alter scientific outputs, conclusions, method Decisions, or validated results; or risk is unclear | Skip Phase 3 → Phase 5 |
| **Impossible** | Root cause cannot be repaired automatically (missing controlled data, Blocking graph issues, unknown failure) | Skip Phase 3 → Phase 5 |

Never treat speculative or unvalidated “fixes” as Safe.

==================================================
Phase 3 — Repair
==================================================

Attempt minimal repairs.

Examples:

- fix paths
- regenerate metadata
- rebuild dependency graph
- regenerate manifests
- repair task references
- repair configuration
- repair environment activation
- regenerate execution commands

Never alter scientific outputs or conclusions.

Also never: overwrite validated results; edit Results/Discussion claims; change Locked **Decision** entries in `method-decision.md` without user direction; delete logs.

Preserve originals when rewriting configs (copy aside or keep prior content in logs). Prefer the smallest change that addresses the diagnosed cause.

==================================================
Phase 4 — Validation
==================================================

Invoke the appropriate verification skill.

Examples:

- verify-todo
- project-auditor
- code-review

Recovery succeeds only if validation succeeds.

Choose the verifier that matches the failure class (e.g. graph/metadata → `@verify-todo`; project consistency → `@project-auditor`; code/config after repair → `@code-review`). Re-run the caller’s expected check when known.

If validation fails, do not claim success — continue to Phase 5.

==================================================
Phase 5 — Escalation
==================================================

If recovery fails:

Generate:

debug-report.md

Include:

- diagnosis
- attempted repairs
- remaining issues
- recommended manual actions

Return failure to the calling skill.

Prefer path: `docs/debug-report.md`. Template: [debug-report-template.md](debug-report-template.md).

Also generate `debug-report.md` (or update it) when recovery is classified Unsafe or Impossible before any repair attempt, so the caller has a clear handoff.

==================================================
Execution Rules
==================================================

- Never fabricate fixes.
- Never modify scientific conclusions.
- Never overwrite validated outputs.
- Always preserve logs.
- Prefer minimal intervention.
- Validate every repair.
- Escalate when automatic recovery is unsafe.

## Return contract

| Outcome | Return to caller |
|---------|------------------|
| Safe repair + validation OK | Success (recovered); optional brief note in caller report |
| Unsafe / Impossible / repair failed / validation failed | Failure + `debug-report.md` |

Do not invoke `@do` or schedule other TODO tasks. Return control to the calling orchestration skill.

## Coordination

| Skill | Role |
|-------|------|
| `@verify-todo` | Invokes on Recoverable graph/metadata issues |
| `@do` | Invokes after monitor escalation / RECOVERABLE before 4.1 retry |
| `@prompt-orchestrator` | Invokes on within-task Safe recovery (Phase 6) |
| `@verify-todo` / `@project-auditor` / `@code-review` | Phase 4 validation |
| `@monitor` | Does **not** invoke `@debug` — escalates to `@do` |

## Artifact registration

If `.cursor/rules/artifact-registry.mdc` is installed: register generated artifacts in `docs/artifact-registry.md` per [artifact-registry-template.md](../_shared/artifact-registry-template.md). If that rule is absent, do not create a registry and do not treat a missing registry as an error.

## Additional resources

- Report template: [debug-report-template.md](debug-report-template.md)
