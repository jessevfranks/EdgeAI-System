# EdgeAI System agent guidance

## Project quality bar

This repository is a small, understandable school/research project. Favor code
that a student can read, explain, and change over code designed for a large
production organization.

- Prefer straightforward functions, modules, dataclasses, and standard-library
  tools.
- Keep control flow visible and names concrete.
- Reuse an existing helper when it genuinely removes duplication, but do not
  introduce layers for hypothetical future needs.
- Avoid service layers, dependency-injection frameworks, factories, plugin
  systems, compatibility shims, and exhaustive defensive checks unless the
  current feature clearly needs them.
- Preserve useful validation at user and file boundaries. Do not mistake shorter
  code for simpler code when it hides behavior or makes errors harder to
  understand.
- Keep comments and documentation focused on intent, assumptions, and non-obvious
  behavior.

## Required final review

After making code changes, the primary implementation agent must spawn the
project-scoped `code_simplifier` agent before reporting completion. Ask it to
review the current diff, apply worthwhile simplifications, and verify the
result. Incorporate its edits and address any concrete issue it reports.

Run this final review once per task, after the implementation is otherwise
complete. Do not recursively spawn another `code_simplifier` from inside that
review.

For documentation-only, configuration-only, or other changes with no executable
code, the primary agent may perform the same brief review directly instead of
spawning the agent.

## Code-change explanation

After the final review and verification, the primary implementation agent must
spawn the project-scoped `change_explainer` agent whenever a task adds or
changes executable code. Give it the original request, the final paths changed
for this task, and the diff base when the changes are not in the working tree.
Wait for its response, then include its concise explanation in the final answer
so the user can understand the feature without reading the entire diff.

Run the explainer once per task, after `code_simplifier`, so it describes the
final implementation. Do not invoke it for documentation-only or
configuration-only work. Do not ask it to edit code, run tests, or perform a
second review, and do not recursively spawn another `change_explainer` from
inside that agent.

## Verification

Use the smallest relevant test command while iterating. Before completion, run
the full normal suite when practical:

```powershell
python -m pytest
```

GPU tests are opt-in and should run only when the task concerns GPU behavior and
the environment is configured for them.
