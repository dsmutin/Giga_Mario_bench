---
name: code-review
description: >-
  Independently review project implementation or recent changes against expected
  behavior and specifications. Use when the user asks for code review, implementation
  audit, correctness check, or review of changes against a prompt or spec.
disable-model-invocation: true
---

# Code Review

## Purpose

Perform an independent review of the current project, implementation or recent changes.

The skill should evaluate whether the implementation satisfies the user's expected behavior, project requirements and supplied prompt without assuming that the implementation is correct.

Follow project rules: **scientific-integrity**, **validation-first**, **reproducibility**, **missing-data-policy**, **method-decision-tracking**.

**Read-only:** Never modify project files. Never automatically fix code.

Methodological truth: prefer recorded decisions in `method-decision.md` (produced/maintained via `@verify-methods`). Never suggest implementations that violate those decisions.

## Inputs

- The current project or selected files.
- An optional description of the expected behavior.
- An optional prompt or specification that should have been implemented.

## Workflow

Copy and track progress:

```
Code review:
- [ ] Step 1: Determine expected behavior (+ read method-decision.md)
- [ ] Step 2: Inspect the implementation
- [ ] Step 3: Compare vs expected behavior and method-decision.md
- [ ] Step 4: Produce structured review
- [ ] Step 5: State review limits (if no issues or always)
```

### Step 1: Determine expected behavior

Establish the review baseline from (in priority order):

1. User-supplied specification or prompt for this review
2. Explicit expected behavior description from the user
3. **`method-decision.md`** — Locked and Tentative methodological decisions (tools, tests, parameters, databases)
4. Project docs: README, specs, `todo.md` task completion criteria
5. If none provided, state assumptions clearly and scope review to **observable defects** (bugs, missing validation, reproducibility gaps) — do not invent functional requirements

If `method-decision.md` is missing, note the gap; do not invent decisions. Method reconstruction belongs to `@verify-methods`, not this skill.

Document the baseline in the review report before judging correctness.

### Step 2: Inspect the implementation

Review:

- correctness
- completeness
- code quality
- reproducibility
- robustness
- maintainability
- documentation
- consistency with project conventions
- **consistency with `method-decision.md`**

Read actual code, configs, and tests. Trace execution paths for critical logic. Check project rules compliance where relevant.

Scope:

- **Full project** — when user requests project-wide review
- **Selected files** — when user points to specific paths
- **Recent changes** — use `git diff` when reviewing a change set

### Step 3: Compare implementation against expected behavior

Identify:

- missing functionality
- partially implemented features
- incorrect behavior
- unnecessary complexity
- duplicated code
- potential bugs
- edge cases
- deviations from the specification
- **contradictions with `method-decision.md`**

#### Method-decision contradictions → Major

Whenever implementation choices contradict recorded entries in `method-decision.md` (different tool, test, parameter, database, threshold, or workflow choice than the **Decision** field):

- Report each contradiction as a **Major** finding
- Cite the method-decision entry and the conflicting code/config path
- **Recommended fix** must restore alignment with the recorded Decision (or explicitly say “user must revise method-decision.md via `@verify-methods` before changing code”)

**Never suggest implementations that violate recorded methodological decisions** — including in Suggestion-severity tips, alternative designs, or “consider switching to …”.

If a newer SOTA alternative is noted in method-decision verification notes, do not recommend adopting it in code until the Decision is updated with user approval.

Do not assume that an implementation is correct simply because it executes successfully.

Verify claims with evidence (file paths, line references, log excerpts, missing outputs).

### Step 4: Produce structured review

Use [review-template.md](review-template.md).

Classify every finding as:

- **Critical** — wrong results, data loss, security, blocks execution
- **Major** — significant spec deviation, missing core feature, reproducibility break, **or any contradiction with `method-decision.md`**
- **Minor** — limited impact bug, incomplete edge handling
- **Suggestion** — improvement not required for correctness (**must not** violate method-decision.md)

For every issue include:

- description
- affected files
- evidence
- expected behavior
- recommended fix

### Step 5: Review conclusion

If no issues are found, explicitly state that the implementation appears consistent with the supplied specification, while noting that the review cannot guarantee the absence of undiscovered defects.

Save report to user path or suggest `docs/code-review.md`.

## Rules

- Review independently.
- Prefer evidence over assumptions.
- Never invent issues.
- Never modify project files.
- Never automatically fix code.
- Focus on correctness relative to the supplied specification rather than coding style alone.
- Report every `method-decision.md` contradiction as **Major**.
- Never suggest implementations that violate recorded methodological decisions.

## Invoked by other skills

| Caller | Scope |
|--------|--------|
| `@do` (step 4.2) | After `@prompt-orchestrator` finishes a task — compare implementation to TODO / acceptance criteria / `method-decision.md` |
| `@prompt-orchestrator` | Within-task when needed for deliverables in the execution plan (not a substitute for do 4.2) |
| `@debug` (Phase 4 validation) | After a Safe repair that touched code/config |

**Not a caller:** `@monitor` (status-only; escalates to `@do`, which may then run `@debug` / return to 4.1 → 4.2).

When reviewing after a job failure (via `@do` / `@debug` context):

- Scope to failing component + log/stack context
- Flag **Critical/Major** that block retry vs operational fixes (resources, env)

## Artifact registration

If `.cursor/rules/artifact-registry.mdc` is installed: register generated artifacts in `docs/artifact-registry.md` per [artifact-registry-template.md](../_shared/artifact-registry-template.md). If that rule is absent, do not create a registry and do not treat a missing registry as an error.

## Additional resources

- Report template: [review-template.md](review-template.md)
