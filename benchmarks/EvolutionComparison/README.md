# US-02 program comparison host

This executable uses the real `ProgramGenome`, `ProgramEvolutionTask`,
`LlmProgramVariationOperator<double>` and `EvolutionEngine<ProgramGenome>`.
It sends model requests and exact candidate source to the common benchmark broker
in AiDotNet.Evolution PR #46. It does not execute candidate code, hold provider
credentials, or install a model API client.

Build once for the final verification batch:

```powershell
dotnet build benchmarks/EvolutionComparison/EvolutionComparison.csproj -c Release -p:UseLocalEvolution=true -p:EvolutionProjectPath=C:/path/to/AiDotNet.Evolution/src/AiDotNet.Evolution/AiDotNet.Evolution.csproj
```

The primary repository's `run_program_comparison.py` launches this executable.
It supplies a loopback-only endpoint and an ephemeral capability through
`EVOLUTION_BROKER_ENDPOINT` and `EVOLUTION_BROKER_CAPABILITY`. Proxies and HTTP
redirects are disabled. Never pass those variables or model credentials into a
candidate sandbox.

The `controlled` track uses a shared task-and-parent-only template. The
`native-bounded` track uses AiDotNet's prompt builder defaults. Both use full
rewrites without proposal retries, sequential evaluation, one island, a 64-cell
source-length archive, and disabled evaluation caching. These are disclosed
benchmark configurations, not claims about untouched production defaults.

The broker independently enforces model/evaluator call limits, verifies the
initial program, and retains exact prompts, responses, and evaluator receipts.
Its receipts—not a process exit code—determine campaign completion. Provider
usage stays in the transport evidence; missing usage is never represented as
zero cost. The fixture campaign does not execute generated code and cannot
establish correctness, speedup, or superiority over OpenEvolve.

Related story: ooples/AiDotNet.Evolution#20. Shared foundation: AiDotNet PR #2148.
Final verification and review evidence must be attached before review readiness.
