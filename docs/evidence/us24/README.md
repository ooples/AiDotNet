# US-24 final validation

Production source: `93b61112c6cf9156b2b6a24db0cd783703ee6bb8`.
Companion Evolution source: `255feb24369702a32ea9db7a3f8a0b7a847d2762`.
The following evidence-only commit adds the compatibility harness and receipts;
it does not change the tested runtime implementation.

[verification.zip](verification.zip): 2,288,944 bytes, 112 readable entries.
SHA-256: `893011b3815d6ff52b0c285eed6ad00fed6a3262f8f8803f6ca64f4fd51e85c9`.
Contains local/hosted TRX and coverage, final build/smoke logs, corrected fixtures,
integrity-checked invocation/comparison bundles, and retained setup/dependency
failures. It is an audit bundle, not a distributable runtime or authenticity seal.

## Results

| Gate | Result |
| --- | --- |
| Windows facade, .NET 10, default analyzers | 1,012 passed, zero skipped/failed |
| Windows facade, .NET 8, default analyzers | 1,012 passed, zero skipped/failed |
| Windows CLI/worker, .NET 10 | 99 passed, zero skipped/failed; builds zero warnings/errors |
| Windows .NET Framework 4.7.1 compatibility | 17 passed, zero skipped/failed; full facade built |
| Linux facade, .NET 10 / .NET 8 | 1,012 passed on each, zero skipped/failed |
| Linux compiler/worker, .NET 10 / .NET 8 | 79 passed on each, zero skipped/failed |
| Linux CLI/worker, .NET 10 | 99 passed, zero skipped/failed |

Hosted source-pinned run: [34885134332](https://github.com/ooples/AiDotNet/actions/runs/34885134332).
Both matrix jobs passed; TRX and coverage attachments are retained in its artifacts.
Facade builds retain existing repository analyzer warnings. This is not a claim
that every repository test or the separate package-path release gate passed.

The Windows authored Python/script smoke completed run → resume → inspect-record →
export → compare. It rejected `print(6)` against expected `7`, despite the fitness
script offering the incorrect candidate quality 100. Four existing numbered
checkpoint-output files remained byte-identical. Resume kept lifetime attempts 2
and invocation-segment attempts 0. Schema-2 configuration templates and live
model/operator/queue fields were present; the manual model queue remained empty.
Winner source SHA-256 across all exports:
`79b9eb38076e36b489114c02fb1cf1b29628e0be7cc91ca76fd121d32e941c60`.

Retained failures explain the verification setup, not hidden passes:

- The initial CLI test command disabled dependency builds before rebuilding the
  CLI; it consumed the older CLI DLL. Explicit CLI/worker builds corrected this
  without rebuilding the facade.
- The initial smoke evaluator omitted its required `evaluate` entrypoint.
  Preflight refused it with exit 3; the corrected fixture used a new output root.
- The earlier correctness-cache red fixture demonstrated that an outer cache hit
  bypassed the fresh gate. Current gated/ungated regression cases pass.
- The focused .NET471 harness initially lacked the existing `AiDotNetTests` friend
  assembly identity. Setting that test-only identity fixed its internal-access
  compile errors; the already-built facade was reused, not rebuilt.

## Reproduction

Use sibling checkouts of these source revisions. Do not replace a NuGet-cache
package with local binaries. For example, from this repository on Windows:

```powershell
$env:EvolutionProjectPath = 'C:/repos/AiDotNet.Evolution/src/AiDotNet.Evolution/AiDotNet.Evolution.csproj'
$env:DOTNET_PROCESSOR_COUNT = '4'
$env:DOTNET_gcServer = '0'
$env:DOTNET_GCHeapHardLimit = '0x300000000'
dotnet test tests/AiDotNet.Evolution.Integration.Tests -c Release -f net10.0 -m:1 -p:UseLocalEvolution=true -p:GeneratePackageOnBuild=false -p:UseSharedCompilation=false
dotnet test tests/AiDotNet.Evolution.Integration.Tests -c Release -f net8.0 -m:1 -p:UseLocalEvolution=true -p:GeneratePackageOnBuild=false -p:UseSharedCompilation=false
dotnet test tests/AiDotNet.Evolution.Compatibility.Tests -c Release -m:1 -p:UseLocalEvolution=true -p:GeneratePackageOnBuild=false -p:UseSharedCompilation=false
```

When reusing a validated facade with `BuildProjectReferences=false`, first build
`tools/AiDotNet.Evolve.Cli` and `tools/AiDotNet.CSharp.Worker` with that property,
then test `tests/AiDotNet.Evolve.Cli.Tests` with it. It skips **all** project-reference
builds, not only the facade. The hosted workflow demonstrates the normal
project-reference build path instead. No paid model service is required.

## Limits and readiness

These are CLI-first lifecycle/inspection/export results, not competitor benchmarks,
an optional dashboard, independent held-out validation, a full environment
attestation or a production isolation boundary. Configuration exports are replay
templates requiring explicit private/runtime bindings. Unknown or encoded secrets
still require review before sharing candidate code.

Resource-ledger/persistent-provider preflight remains explicitly unsupported; it
does not silently bypass reservation/checkpoint restrictions. Custom runner queues
are available only when the runner implements the telemetry contract.

Ordinary package-path website/wiki/sample CI reports missing unpublished Evolution
APIs. Merge/package readiness requires the companion dependencies to be delivered;
passing the pinned-source gate does not waive that requirement. No merge, package
publication, deployment or completion of the 26-story roadmap is asserted.
