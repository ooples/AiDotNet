# Evolution CLI — US24 implementation in progress

This repository tool is not added to the core NuGet dependency graph. US24 is
**not complete**: executable seed preflight, durable inspection/export bundles,
provider/model telemetry, backend queues and optional dashboard work remain.

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

The full .NET 10 facade integration suite passed **978 tests**, with no skips, using
the built AiDotNet assembly and explicit local Evolution US10 source dependency
`255feb24369702a32ea9db7a3f8a0b7a847d2762`. New tests cover batch drain, 2→4 evaluation
checkpoint resume, trace delivery despite observer failure, fatal exception
preservation, parsed configuration snapshots, checkpoint winner files and sanitized
output failure reporting. The actual CLI/worker project suite passed **62 tests**
with no skips. Tests are maintained in `tests/AiDotNet.Evolve.Cli.Tests`; the lifecycle test sends a real
local pipe pause during an evaluation and resumes the resulting engine checkpoint.

This is local implementation evidence, not hosted CI approval, package publication,
cross-platform pipe certification, competitor superiority or completion of US24.
