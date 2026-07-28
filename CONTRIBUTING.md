# Contributing

Contributions should preserve the traceability of RiverSort AI results and the
separation between domain logic, application orchestration, adapters, and UI.

## Local setup

```bash
git clone https://github.com/b1n0r10n/riversort-ai.git
cd riversort-ai
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Workflow

1. Create a branch from the latest `main`.
2. Keep the change focused and add tests for changed behavior.
3. Run the checks below.
4. Review the staged diff for data, weights, credentials, and local paths.
5. Push the branch and open a pull request.

```bash
python -m pytest -q
python -m pip check
python scripts/release/audit_public_repository.py
git diff --check
```

Do not force-push or delete `main`. Revert merged changes with a new commit.

## Design constraints

- Domain modules use the Python standard library only.
- The presentation layer calls application services, not model libraries.
- Model, tracker, media, priority, and recommendation values come from
  versioned configuration.
- Recommendation output must include its observed value, operator, threshold,
  reason, and action.
- Unsupported target classes use `not_supported` and null metrics.
- Empty detections must not be described as a clean river.
- Changes to taxonomy, schema, formulas, thresholds, or evaluation claims need
  tests and corresponding documentation.

## Data and artifacts

The repository may contain source code, configuration, tests, documentation,
small legal fixtures, checksums, model cards, and numeric evaluation reports.

Do not commit:

- dataset images or extracted archives;
- user media or sensitive metadata;
- model weights or experiment directories;
- virtual environments and caches;
- credentials, private URLs, or machine-specific paths.

Model weights are distributed as release assets and verified by SHA-256.

## Pull requests

A pull request should explain the problem, the behavior change, validation
performed, compatibility impact, and any remaining limitation. Visual changes
should include screenshots that are safe to publish.
