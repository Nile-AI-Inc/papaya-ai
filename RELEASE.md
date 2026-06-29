# Papaya AI SDK Release Checklist

This package is public OSS so customers can inspect the wrapper they embed in production agent services.

## Preflight

- Confirm the release branch contains only intentional SDK/package/docs changes.
- Confirm `package.json` has the intended version, license, repository, files list, engines, and provenance settings.
- Confirm README starts with the shortest setup path, then documents fetch-based custom agent loops, provider SDK wrappers, workflow boundaries, capture modes, and safety defaults.
- Confirm examples use environment variables for provider keys and Papaya tokens.
- Confirm no secrets, customer trace bodies, local paths, or internal deployment tokens are present in README, tests, package metadata, workflows, or generated artifacts.

## Required Checks

Run the TypeScript checks from the standalone SDK repo root:

```sh
npm install
npm run typecheck
npm test
npm pack --dry-run
```

Run the Python checks from the standalone SDK repo root:

```sh
python -m pip install --upgrade pip build twine
python -m unittest discover packages/papaya-ai-python/tests
python -m build packages/papaya-ai-python
python -m twine check packages/papaya-ai-python/dist/*
python -m pip install "$(find packages/papaya-ai-python/dist -name 'papaya_ai-*.whl' -print -quit)[langchain]"
python -c "from papaya_ai import Papaya; from papaya_ai.integrations.langchain import PapayaCallbackHandler; print(Papaya.__name__, PapayaCallbackHandler.__name__)"
```

When validating from the Supernova monorepo, run:

```sh
npm run typecheck --workspace @papaya-ai/tracing
npm run test --workspace @papaya-ai/tracing
npm pack --workspace @papaya-ai/tracing --dry-run
npm run test:papaya-ai-sdk-release
python -m unittest discover packages/papaya-ai-python/tests
python -m build packages/papaya-ai-python
python -m twine check packages/papaya-ai-python/dist/*
```

The npm dry-run package contents must include only:

- `dist/**`
- `README.md`
- `RELEASE.md`
- `LICENSE`
- `package.json`

The Python package contents must include only:

- `papaya_ai/**`
- package metadata

## Release Evidence

Attach these to the release notes:

- Package version.
- Git commit SHA.
- Output summary from the required checks.
- Dry-run package file list.
- Python wheel and sdist metadata check output.
- Hosted ingest smoke result for `https://papaya.fyi/api/v1/ingest/traces`.
- Any known limitations, especially provider coverage and the current explicit `flush()` behavior.

## Publish

Publish with npm provenance from the guarded GitHub Actions publish job after npm Trusted Publishing is configured for `@papaya-ai/tracing`:

```sh
gh workflow run npm-publish.yml -f publish_npm=true
```

If publishing from the Supernova monorepo before the standalone repository is the source of truth, use:

```sh
gh workflow run papaya-ai-sdk.yml -f publish_npm=true
```

Publish the Python SDK from the guarded GitHub Actions publish job after PyPI Trusted Publishing is configured for the `papaya-ai` project or pending publisher:

```sh
gh workflow run npm-publish.yml -f publish_pypi=true
```

Publish a stable version only after a hosted production smoke has passed.

## Post-Publish Smoke

Install the published npm package in a clean test app and verify:

- Existing provider calls still receive no `papaya` field.
- `capture=metadata` exports payload metadata without values.
- `capture=redacted` redacts obvious emails, tokens, API-key-like values, and sensitive object keys before export.
- `papaya.run()` groups multiple provider calls under one run.
- A real hosted ingest token receives a 202 trace smoke through `https://papaya.fyi/api/v1/ingest/traces`.

Install the published Python package in a clean virtual environment and verify:

```sh
python -m venv /tmp/papaya-python-smoke
/tmp/papaya-python-smoke/bin/python -m pip install --upgrade pip
/tmp/papaya-python-smoke/bin/python -m pip install "papaya-ai[langchain]"
/tmp/papaya-python-smoke/bin/python -c "from papaya_ai import Papaya; from papaya_ai.integrations.langchain import PapayaCallbackHandler; print(Papaya.__name__, PapayaCallbackHandler.__name__)"
```
