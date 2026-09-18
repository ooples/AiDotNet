# Authored C# runtime pilot

This is an executable development workflow for roadmap US-01/03/05/11/24, not their completion or an OpenEvolve comparison.
It uses no model provider and no paid services. It proposes two fixed authored programs, not AI-generated improvements.

## Run

Build the CLI and worker with a compatible Evolution dependency. Until the companion core changes are published,
use the documented local project switch with the matching checkout; the old published preview is not sufficient.

```powershell
dotnet build tools/AiDotNet.Evolve.Cli/AiDotNet.Evolve.Cli.csproj -c Release -p:UseLocalEvolution=true
dotnet build tools/AiDotNet.CSharp.Worker/AiDotNet.CSharp.Worker.csproj -c Release -f net10.0
dotnet tools/AiDotNet.Evolve.Cli/bin/Release/net10.0/aidotnet-evolve.dll benchmark-program --worker C:/absolute/checkout/tools/AiDotNet.CSharp.Worker/bin/Release/net10.0/AiDotNet.CSharp.Worker.dll --output C:/absolute/new-evidence-directory --runs 4 --measurements 3
```

Use a trusted worker built from this checkout. The command refuses non-catalog source, relative worker paths,
shell/template metacharacters and an existing output directory. Runs are bounded to 1–12 and measurements to 1–9.
The process supervisor is **not an OS security sandbox**. Do not adapt this runner to hostile generated code without
an independently enforced filesystem/network/identity boundary. The worker's 256 MiB managed GC cap is not an RSS
limit; the CLI validates that configured cap in the bounded runtimeconfig. The supervisor attempts a 4096 MiB
memory limit and 30-second timeout, without attesting OS enforcement. Reports identify the controller runtime and
the worker's requested framework separately; runtime resolution inside the child is not independently attested.

## What is measured

Each run starts with recursive Fibonacci, proposes an intentionally incorrect implementation, then proposes an
iterative implementation. An independent fast-doubling reference checks public inputs 0, 1, 10 and 20 before timing.
The wrong candidate must not reach search timing or promotion. All correct candidates receive the same number of
search measurements at input 39. The archive minimizes median elapsed milliseconds.

After search, the frozen winner and original baseline are measured at public input 40; their order alternates across
runs. Confirmation never changes the selected winner. These are public development checks, not sealed generalization
tests. A baseline can legitimately remain the winner; the tool does not manufacture a speedup or retry until one wins.

The host stopwatch surrounds a fresh worker request, including startup, Roslyn compilation, candidate execution
and cleanup. It does **not** isolate algorithm time. There is no warmup subtraction, CPU affinity, controlled host
load, prospectively powered experiment, cross-family validation or significance claim. Repeated calls are nested
within runs, not independent tasks. Do not interpret an apparent ratio as evidence of competitive superiority.

One `authored-csharp-worker-dispatch-v1` unit means one dispatched worker request, including failures and cancellation.
The shared ledger covers correctness, search and confirmation, reserving maxima before evaluation. Missing receipts
retain the full reservation. Compilation inside a worker request is not separately metered; no dollar, CPU or RAM
cost is inferred from these units. Every run's maximum is `3 * (4 + measurements) + 2 * measurements` units.

The write-once `plan.json` is created before dispatch. `run-00.json`, subsequent run files and `report.json` retain all
planned runs, including failed/not-started runs, every raw sample, candidate identity/source, confirmation order,
search result and ledger receipts. Files use create-new pending writes and non-overwriting moves. A hard kill may
leave only the plan, completed run files or a pending write; it is not resumable. Binary hashes before and after the
run detect replacement of listed dependencies, not a hostile host or every OS/runtime dependency. Local absolute
binary paths are included; review evidence before sharing it. No raw subprocess exception payload is retained.

## Bring your own fitness service

As an algorithm author, I want runtime scoring through the public facade so correctness and accounting still apply.

Given a caller-owned `IProgramFitnessEvaluator`, when it is assigned to `ProgramEvolutionOptions.CustomFitnessEvaluator`,
then no test cases, scoring script or facade execution engine are required. Custom proposals still need
`CustomVariation`, or a chat client for the normal proposal loop. `ConfigureProgramCorrectness` remains independent.

Given custom fitness and either `TestCases` or `EvaluatorScript`, when configuring a run, then validation rejects the
ambiguity. Given an evaluator whose `Id` or `VersionHash` changes before or during evaluation, then its result cannot
be promoted; a dispatched call without a trustworthy receipt retains its resource reservation.

Given a minimization evaluator, when configuring the archive, then use `ConfigureEvolution(new EvolutionOptions
{ ArchiveDirection = EvolutionOptimizationDirection.Minimize, ... })` with the complete desired search settings.
`ConfigureEvolution` settings take precedence over `ProgramEvolutionOptions.Engine`; the default archive maximizes
and rejects mismatched directions. The tool configures both consistently.

The backend must version workload, reference data, runtime, measurement protocol and scoring semantics; unchanged
declared identities cannot prove unchanged hidden implementation/data. It owns isolation, timeouts and truthful
same-unit costs. Configuration clones share the backend; the builder neither disposes nor serializes it. Use fresh
instances for independent runs. Coordinated resource-ledger/engine checkpointing remains unsupported.

## Acceptance evidence

As an experiment owner, I want failures represented rather than dropped from favorable timing summaries.

Given a planned run, when execution is cancelled or fails, then the report preserves its status, dispatched samples
and resource receipts; other planned runs remain visible. Given invalid/truncated output, when scoring runtime, then
the candidate cannot receive completed timing fitness. Given unexpected executor failure, then the sample remains
unknown and the ledger retains the maximum. Given an existing result file, then another write cannot replace it.

The focused `tests/AiDotNet.Evolve.Cli.Tests` project exercises these contracts, existing CLI commands and the real
worker/facade path without a live provider. It makes no timing-order assertion. The compiler-guided workflow runs
the suite on .NET 10; the custom fitness library surface also targets .NET 8 and .NET Framework 4.7.1.

### Local verification, September 10–11, 2026

- 901 focused consumer tests pass separately on .NET 8 and .NET 10, including 13 new custom-fitness cases.
  Production/test-copy DLL hashes match. The version-pinned wrapper has 100% line/branch coverage; the selected
  program/options paths have 5,026/5,440 lines and 2,456/3,011 branches covered. That selection excludes the builder
  and unrelated AiDotNet types. TRX/coverage: `TestResults/custom-fitness-final-net8` and `custom-fitness-final-net10`.
- 79 optional compiler/worker tests pass separately on both modern frameworks against the updated consumer
  (`TestResults/custom-fitness-compiler-net8` and `custom-fitness-compiler-net10`). Compilation/worker execution is
  real; chat responses are scripted. This does not repeat or supersede the earlier optional-package coverage run.
- 42 CLI tests pass on .NET 10, including the real worker pilot, configuration validation, cancellation, failure
  accounting and write-once evidence. `ProgramBenchmark.cs` has 268/270 executable lines covered; the full CLI module
  has 389/463 lines and 229/282 branches covered. The workflow gates the new benchmark file at 90%, not the entire
  existing CLI. Coverage: `TestResults/runtime-cli-attested-config`; that directory name refers to validated
  configuration, not attested runtime isolation. The gate rejects missing, wrong-module and below-threshold fixtures.
- Explicit library rebuilds succeed on net8.0, net10.0 and net471 with zero errors. Diagnostic analyzers were disabled;
  existing warnings remain. No .NET Framework test suite, whole-repository validation or current-head hosted approval
  is claimed. The source-checkout CI layout fix and normal compatible NuGet dependency path need separate validation.

Coverlet 10.0.1 rejects literal hyphens in include filters; `[aidotnet-evolve]*` was discarded and a local collector
started scanning unrelated dependencies before being stopped without running tests. `[aidotnet*evolve]*` selects
the actual CLI module, which the coverage gate checks explicitly. The version-pinned
[Coverlet filter implementation](https://github.com/coverlet-coverage/coverlet/blob/d21b5b6a08d48f51405ba2c5c5660f91a565776d/src/coverlet.core/Helpers/InstrumentationHelper.cs#L336)
explains the rejected filter and fallback. The stopped collection is not counted as a test pass or test failure.

### Retained development pilot

The [four-run raw report](benchmarks/authored-runtime-2026-09-11.json) retains all 96 dispatched samples and 96 worker
units, zero unknown receipts, all four failed-correctness proposals and all successful confirmations.
Each run proposed the same incorrect source once; its four failed checks remain represented.
All listed binary hashes were unchanged. The controller was .NET 10.0.12 on Windows X64; the worker requested
Microsoft.NETCore.App/10.0.0. No other builds/tests from this task ran concurrently; other host load was uncontrolled.

Per-run confirmation medians (baseline / selected iterative implementation, milliseconds):
`1625.6424 / 1305.2734`, `1631.6772 / 1289.9025`, `1625.9062 / 1383.8596`, `1746.7493 / 1327.8348`.
These numbers describe this public authored end-to-end pilot only—not isolated Fibonacci runtime, an LLM's ability
to discover an optimization, held-out generalization, statistical significance or superiority to OpenEvolve.

The committed export removes local absolute binary paths and normalizes JSON formatting; it does not remove runs
or samples. Original report SHA-256: `7fa3e8d6bcc6bf41a3dc7fbae536b946cf67535f51b9b730925a617c8ef8eac7`.
