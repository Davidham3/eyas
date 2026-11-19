# Plugin backend — your inference engine, registered from outside the package

The backend counterpart to [`examples/strategy_skeleton/`](../strategy_skeleton/):
one file that implements `InferenceBackend` and registers it with
`@register_backend`, plus the config and the CI test that prove the
registration route works for **backends** — the same route built-ins obey,
exercised from the outside. Nothing here needs a GPU, and nothing in the
framework (`configs/`, `eyas.builtin`) names `echo_demo`: the name exists only
once your config (or your package metadata) registers it.

## Contents

| File | What it is |
| --- | --- |
| `echo_backend.py` | The backend: `@register_backend("echo_demo")`, one deterministic `agenerate` |
| `echo_backend_test.py` | The CI gate: registration via `plugins`, order/echo contract, `close()`, and the run through the real entry |
| `demo_config.yaml` | A runnable config: the example backend on the zero-data `demo_shapes` task with the built-in `raw` method |

## Run it

From the repo root (the `plugins` key resolves dotted paths against the CWD,
which is why every command here runs from the root):

```bash
uv run eyas run --config examples/plugin_backend/demo_config.yaml --output-dir .eyas-demo/plugin-echo
uv run eyas status .eyas-demo/plugin-echo   # or cat its result.json
```

Any scratch `--output-dir` works (a throwaway `.eyas-demo/` in the checkout,
or a path under `/tmp`); it is demo output — delete it when you are done.
Without the flag, the run lands under `outputs/` like any other run.

## The two registration routes

**Route 1 — the config `plugins` key (unpackaged code, what this example
uses).** The module path is imported before any registry lookup, and its
decorator does the registering — no install step, no fork of the framework:

```yaml
plugins: [examples.plugin_backend.echo_backend]

backend:
  name: echo_demo # resolvable because the line above imported it
```

**Route 2 — entry points (packaged plugins).** Ship the backend in a
distribution and advertise it; no `plugins` line is needed anywhere — the
framework discovers and imports it at startup:

```toml
# pyproject.toml of your own package
[project.entry-points."eyas.plugins"]
my_plugins = "my_pkg.eyas_plugins" # a module whose import performs the registration
```

Both routes end at the same registry. Verify route 2 with a bare
`uv run eyas list` (entry points load automatically); for route 1, run
`uv run eyas list --config <your config>`, which imports the `plugins:` modules
before listing.

## Rules of the road

The full contract is [`docs/tutorials/backends.md`](../../docs/tutorials/backends.md)
("Adding your own backend"); the parts this example exercises, and the parts
your engine must not skip:

- Return **one response per request, in input order**; never raise for a single
  bad request if the batch can continue — encode per-request failure in the
  response (that is what `finish_reason` is for).
- Route sync calls through the provided `generate()` helper — it uses the
  framework's process-resident event loop; calling `asyncio.run` yourself will
  deadlock in-process engines.
- Bump the activity watchdog (`eyas.backends.vllm` shows the pattern) on every
  engine interaction if you implement long-blocking calls — a hung GPU job is
  worse than a dead one, and the watchdog kills anyone who looks like one.
- Override `close()` if you hold engines/processes/connections — the run calls
  it after the loop so the next run in a sweep does not pay your GC bill
  (`EchoBackend.closed` is the example's observable proof that this hook fires).
- Request messages hold **typed parts**, not strings: `user_message(ImagePart(...),
  TextPart(...))`. Filter to `TextPart` when reading text (as `EchoBackend` does) —
  the same traversal `eyas.backends.base.user_text` uses.

## Run the tests

```bash
uv run pytest examples/plugin_backend -q
```

The repo gate collects this directory automatically (`testpaths` in
`pyproject.toml` includes `examples`), so the example cannot rot silently. The
test module deliberately **fails loudly** (a named `ImportError`, never a skip)
when the installed `eyas` distribution is too stale to contain the features it
pins — the remedy is printed with the error.

## Take it further

Swap `name: echo_demo` in `demo_config.yaml` for any registered backend and the
rest of the config is untouched — that config-level interchangeability is the
framework's backend-agnosticism contract (docs/tutorials/backends.md).
