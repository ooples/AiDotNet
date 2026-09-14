# Evolution CLI — US24

This repository tool stays outside the core NuGet dependency graph. It supports
configured program adapters, local inspection/control, and immutable invocation
exports. The optional dashboard is not required to use the CLI. Unsupported
coordinated-provider preflight is refused explicitly; this is not a cluster UI,
an independent held-out validation service, or proof of better optimization.

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

Fresh-correctness gates also disable the **outer engine evaluation cache**, even
when requested in engine settings: a cache hit would otherwise bypass the checker
and its identity guard. Repeated canonical candidates become `Duplicate`, not
another cached `Completed` result. Ungated runs retain their configured caching.
Evidence-validated performance reuse inside a provider can still operate behind a
fresh checker. The changed engine cache policy is checkpoint-incompatible with
earlier cached gated runs; do not relabel or force-load those checkpoints.

The preflight allowance is **separate from** `MaxEvaluationAttempts`: at most one
correctness evaluation and one additional fitness evaluation. A built-in evaluator
may dispatch once per public test, subject to its configured runner limits. Opaque
custom evaluators must enforce their own resource limits. `EvaluationTimeout`
requests cancellation per preflight stage. With `EvaluationGracePeriod`, the tool
stops waiting after timeout plus grace, including for a synchronously blocking
provider. Abandoned work may still execute: its consumption is unknown, no further
stage/search starts, and owned runners are retained until that work settles.
Without grace, cancellation is cooperative. These controls are not an API spending
cap. The search's `TimeLimit` does not include the separate preflight allowance.
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

Cooperating CLI processes hold exclusive file leases on output, checkpoint and
explicit trace directories before preflight. A competing writer fails without
starting search. This is not a security boundary against a noncooperating process
or an attacker changing filesystem links. Leases are released on disposal/process
exit; their empty files remain to avoid unlink races. CLI-derived output roots are
retained, not automatically recursively deleted. Choose a new root to start an
independent experiment.

Checkpoint **output snapshots** are now write-once source/info pairs. A new observer
continues the highest existing output ordinal; these ordinals are not the engine's
checkpoint safe-sequence numbers. Explicit attempts to replace an existing snapshot
fail. Existing history is scanned without following subdirectories, up to one
million entries; beyond that, select a new output root. Final `best/` files remain
mutable convenience outputs. `MaxRetainedRecords` bounds recent in-memory write
receipts (default 1,024; range 1–65,536), and `DroppedRecords` exposes evictions.
This does not delete any historical files. Failed staging directories are retained
for diagnosis; filesystem capacity still needs operator management.

## Invocation records and exports

```text
run --config experiment.yaml --record records/attempt1 --include-source --json
inspect-record --record records/attempt1
export --record records/attempt1 --out exports/attempt1
compare --left records/attempt1 --right records/attempt2 --out comparisons/pair1
```

`--record` is opt-in and requires a **new** destination for every invocation,
including resumes. The destination is leased before preflight. Records include
unsuccessful preflight, cancellation and handled search failure; abrupt process
termination cannot promise a final record. The command returns a failure if a
successful run cannot publish its requested record. No existing record is replaced.

Each bundle contains a SHA-256 manifest plus `configuration.json`,
`environment.json`, `validation.json`, `result.json`, and optionally `program.txt`.
Metadata is bounded to 128 KiB per file and source to 4 MiB. Reads reject duplicate
JSON properties, unknown/traversing names, file-size violations, linked roots/files,
hash/length mismatches and disagreement between source and winner receipts. Staging
and same-parent publication keep a partially written bundle from appearing at the
requested destination. Unix staging uses owner-only permissions; Windows inherits
the parent ACL. Neither is an authorization system or authenticity signature.

Configuration is a **structured replay template** of parsed settings and CLI
overrides, including evaluator, sandbox, descriptor and search settings. Private
paths, run IDs, parameter bags, environment and runtime services become explicit
`RequiresBinding` placeholders; `RequiredBindings` identifies their JSON pointers.
Without `--include-source`, prompts, seeds, examples and evaluator script also
become placeholders with content hashes. No credential hashes are published.
Restore required bindings before replay; never send placeholders to a provider.
Derived facade defaults remain tied to the recorded facade binary, not separately
serialized. Metadata bounds and known-secret checks run before expensive search.
Environment evidence includes
runtime/platform/architecture and exact CLI/facade/core binary hashes, checking
loaded module identity against disk first. It does not fingerprint every transitive
dependency, sandbox, device or external evaluator. `validation.json` retains seed
preflight and archive receipts, **not independent held-out winner revalidation**.
Provider-reported final usage is process-segment scoped; zero tokens may mean the
provider did not report usage, and no currency total is inferred.

Source is omitted unless `--include-source` is supplied. When included it is taken
from the in-memory winner, without truncation or rewriting, and must match the
archive receipt. Configured parameter-bag and credential/environment strings are
treated conservatively as sensitive: finding one in source or decoded JSON blocks publication. This
can reject benign strings too. It cannot detect arbitrary unknown or encoded
secrets. **Review source before sharing it**; no credentials are loaded from global
environment/credential stores for this scan. The export command copies a verified
bundle, including any source already explicitly captured; it is not a new secret
scanner. Comparison export retains both independently verifiable bundles and a
descriptive report with a root manifest. Equal configuration/environment
templates do not establish equal tasks, fair budgets or statistical significance.
Hashes attest integrity, never scientific validity or competitor superiority.

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
output. `inspect-record` provides durable historical inspection when `--record`
was requested; it cannot reconstruct an unrecorded or abruptly terminated run.

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
  4,096 identities. This is not the evaluator's backend queue.
- `Runtime` reports operator/configured-model identities, proposals, chat calls,
  retries, abandoned proposals, provider errors, and reported token subtotals. Labels
  are hashed (`sha256:` plus SHA-256 for non-hash labels); correlate them with your
  private configuration. Configured identity is not per-response routing evidence.
  Unreported tokens are null after chat calls, never an assertion of free usage.
- Built-in runners expose waiting and active executions separately, including
  cancellation while queued. Custom runners can implement
  `IProgramExecutionTelemetrySource`; otherwise queue values are null. These are
  per-instance counts, not OS/cluster queues. Custom variation usage getters must
  be thread-safe and nonblocking. Final inspection freezes a detached sample.
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
before configuration validation. Program output remains opt-in; use `--record`
for integrity-checked, bounded exports rather than mutable convenience outputs.

## Verification

Production revision `93b61112c6cf9156b2b6a24db0cd783703ee6bb8` passed the final
default-analyzer gate: **1,012 facade tests on each of .NET 10 and .NET 8**, **99
actual CLI/worker tests**, and **17 focused .NET Framework 4.7.1 compatibility
tests**, zero skips/failures. CLI/worker builds had zero warnings/errors; facade
builds retain baseline repository analyzer warnings. The explicit Evolution source
dependency was `255feb24369702a32ea9db7a3f8a0b7a847d2762`.

The source-pinned hosted workflow passed on Linux too: 1,012 facade and 79 compiler
tests per framework, plus 99 CLI tests. Real pipe tests drain a running batch and
resume its checkpoint; the final Python/script smoke rejects the wrong high-scoring
candidate and preserves historical snapshot files, lifetime budgets and exact
winner bytes through resume/inspect/export/compare, with no model requests.

See [raw evidence, reproduction and limits](../../docs/evidence/us24/README.md).
Initial fixture/build-order failures are retained. Ordinary package-path CI still
depends on unpublished companion APIs: these results establish CLI review
readiness, not merge/package readiness, scientific superiority or completion of
the full roadmap. The optional dashboard remains separate.
