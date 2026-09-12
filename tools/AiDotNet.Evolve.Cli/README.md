# Evolution CLI — US24 implementation in progress

This repository tool is not added to the core NuGet dependency graph. US24 is
**not complete**: coordinated-provider preflight, durable inspection/export bundles,
provider/model telemetry, backend queues and optional dashboard work remain.

## Seed preflight and budgets

`preflight --config experiment.yaml` executes setup checks and the first seed only.
`run` performs the same preflight before search. Both accept
`--preflight-max-tests <1..4096>` (default 256 public input/output cases).

Preflight validates positive search budgets, configured proposal/evaluation
services, source bounds, output write access and seed archive placement. Resume
also checks the checkpoint file's integrity; full compatibility is still verified
by the engine before search. It makes no proposal request or model health-check
call. It does execute the configured seed/evaluator, whose runner must provide an
appropriate containment boundary.

`Completed` alone does not pass correctness: the seed must achieve maximize-one
with no violations or declared reused measurement. Input/output fitness can supply
that check; scripts and custom fitness require explicit correctness or public
input/output examples (custom fitness cannot be combined with examples under the
existing options contract). A seed with the wrong fitness direction or missing
archive coordinates is also refused. This checks public examples, not a held-out
suite. When examples are configured, the CLI requires every candidate to pass
them before script fitness runs. Built-in pass-fraction fitness becomes a hard
gate without executing the examples twice. Library callers opt into this with
`WithProgramTestCaseCorrectness()`; custom providers use a separate
`ConfigureProgramCorrectness` checker. Both correctness and fitness stages are
version-pinned before and after each call. Declared reused correctness never
passes. The gate identity is now `correctness-gated-program-v4-fresh-pinned`;
checkpoints created with older gate identities are not compatible.

The preflight allowance is **separate from** `MaxEvaluationAttempts`: at most one
correctness evaluation and one additional fitness evaluation. A built-in evaluator
may dispatch once per public test, subject to its configured runner limits. Opaque
custom evaluators must enforce their own resource limits. Preflight is not an API
spending cap or enforcement of the search engine's evaluation grace-period policy.
The report keeps correctness and additional fitness costs separate; arbitrary
providers may use different units. Do not add unlike units or interpret them as
money. An exception without a receipt leaves consumption unknown.

Resource-accounted/persistent-fitness preflight is explicitly refused until
coordinated reservations are implemented. There is no silent bypass. A failed
preflight returns exit 3 without starting search; token cancellation returns 2.
Search fail-fast/no-candidate outcomes and runs without a usable archive also
return 3. A successful graceful stop remains distinct from token cancellation.

## Live control

Build this project and invoke its `aidotnet-evolve.dll` with `dotnet`:

```text
run --config experiment.yaml --output experiment-results --session experiment1
inspect --session experiment1
pause --session experiment1
cancel --session experiment1
run --config experiment.yaml --output experiment-results --resume --session experiment1
```

Use the same run ID, compatible configuration and **lifetime** evaluation budget
for resume. Configure `evolution.checkpointInterval` to enable checkpoints. A pause
request drains the current batch and exits; it is not an in-memory suspension.
`stop-requested` is not a checkpoint receipt. The final snapshot says `paused` only
after a graceful stop and a matching, integrity-checked checkpoint have been read.
Without a usable checkpoint the final state is `stopped`, not resumable. Snapshot
verification does not attest that the file cannot change later; resume verifies it
again. Automatic resume remains unsupported for coordinated resource-ledger and
persistent-fitness runs; the tool does not bypass the facade's existing guards.

On `run`, Ctrl+C first requests a graceful stop, a second interrupt cancels and may
roll back in-flight work, and a third allows process termination. Other commands
cancel on the first interrupt. Cancellation is distinct from a successful pause.

The live endpoint is a local, current-user-only named pipe. Names allow 1–64 ASCII
letters, digits, hyphens or underscores. Duplicate live names fail before search.
Only `inspect`, `pause` and `cancel` are accepted; no arbitrary method, path or shell
execution is exposed. Requests are limited to 32 bytes, responses to 64 KiB, and
stalled requests expire after three seconds. This is not protection from another
process running as the same user. Run in an independently provisioned containment
boundary for hostile candidate code; a worker process alone does not restrict its
filesystem or network access.

Inspection exists while the command is running. With `--session`, its final
snapshot is emitted to standard error; `--json` keeps the result summary on standard
output. Durable historical inspection is not implemented yet.

## What progress means

- Archives are copied only during serialized engine callbacks. IPC readers never
  enumerate a live archive. Island occupancy is shown for at most 128 islands.
- Best feasible means archive-accepted; it is **not** a held-out correctness
  certificate. Sixteen recent lineage records retain at most eight parents each.
  Non-hash adapter/operator labels are hashed to avoid exporting private labels.
- Attempts, terminal candidates, statuses and reported evaluation costs cover the
  **current process segment**, not the restored run's lifetime. They include losing
  candidates; archive-only cost sums would omit them. An aborted segment marks
  unknown consumption rather than reporting zero abandoned cost.
- Pending candidates means observed `Proposed` without `Evaluated`, bounded to
  4,096 identities. This is not the evaluator's backend queue; unavailable backend
  queue depth, model identity, token usage and API currency are explicitly absent.
- A checkpoint's existence alone never establishes resumability. The observer
  requires a checkpoint event after the latest terminal evaluation and verifies its
  saved sequence, compatibility hash and integrity before recording its file hash.

## Facade fixes supporting the tool

`WithEvolutionControl` connects a caller-owned, single-run graceful-stop handle.
`ObserveProgramEvolution` combines caller callbacks with built-in observers; a
failure cannot suppress tracing or mask a fatal failure from another observer.
`IProgramEvolutionArchiveObserver` registers island views before execution.
`FromConfiguration` applies one parsed YAML document without reloading the file or
resolving environment variables a second time.

Checkpoint-time program output now sees the archive during execution, uses the
effective output directory, and reports `program_output_incomplete` without private
filesystem diagnostics. `run --output` also synchronizes the program output root
before configuration validation. Program output remains opt-in. This does not yet
provide a credential-reviewed export bundle.

## Current verification

The full .NET 10 facade integration suite passed **989 tests**, with no skips, using
the built AiDotNet assembly and explicit local Evolution US10 source dependency
`255feb24369702a32ea9db7a3f8a0b7a847d2762`. New tests cover batch drain, 2→4 evaluation
checkpoint resume, trace delivery despite observer failure, fatal exception
preservation, parsed configuration snapshots, checkpoint winner files and sanitized
output failure reporting. The actual CLI/worker project suite passed **71 tests**
with no skips. Tests are maintained in `tests/AiDotNet.Evolve.Cli.Tests`; the lifecycle test sends a real
local pipe pause during an evaluation and resumes the resulting engine checkpoint.

The original 978-test control slice used a default analyzer-enabled library build.
The later preflight iteration used `RunAnalyzersDuringBuild=false`; final default-
analyzer and cross-target validation remain required. Initial compiler failures
and the earlier incorrect-seed behavior were retained, not discarded.

An actual CLI/Python smoke used `print(6)` against expected `7`. The old CLI exited
0 and saved that wrong program with quality 0. With preflight it exits 3, reports
one correctness cost unit and writes no winner. `print(7)` passes, records its
separate one-unit preflight, and completes a one-evaluation search. Both use an
empty manual model queue: no model response or paid API call is needed.

This is local implementation evidence, not hosted CI approval, package publication,
cross-platform pipe certification, competitor superiority or completion of US24.
