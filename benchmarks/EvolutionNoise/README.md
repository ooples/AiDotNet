# US-06 consumer noise validation

This study runs the actual AiDotNet `RidgeRegression<double>` training/prediction implementation and two trusted
sorting algorithms through `ProgramNoiseEvaluationSession`. Program genomes select allowlisted configurations;
no generated source is compiled or executed. It requires companion Evolution PR #49 and retains the shared foundations.

## Prospectively fixed design

Roots: 1103, 2207, 3301, 4409, 5519, 6607. No optional stopping or threshold fitting.

- Ridge: independently regenerated training inputs uniform [-1,1], targets `2*x + uniform[-.2,.2]`,
  8 rows for cheap screening, 64 for search/confirmation, fresh 64-row test draws per fit. Every callback creates
  a new `RidgeRegression<double>` with no intercept and alpha in {0,4,10000}. Fitness `1/(1+MSE)` has declared
  support [0,1]. Correctness checks configuration admissibility; finite predictions are checked after training.
- Two declared ridge screen thresholds, .5 and .98. Two screen replicates; four full search and 128 confirmation
  replicates per candidate. Full-fidelity usefulness threshold .7; minimum confirmed improvement .05;
  two challenge slots at 95% family confidence. Audit up to three rejects (a census for this candidate set).
  Challenge alpha0 against alpha10000, then the reverse as a negative control.
- Sorting: fresh 128-element screen inputs, 2048-element full inputs; Array.Sort versus insertion sort.
  Every invocation copies the original input and verifies output against the trusted sorted reference.
  One explicit warmup plus one measured invocation per observation. Support [0,1000]ms with overrun failure,
  minimization, 32 confirmation/audit replicates. Zero-ms cheap threshold is deliberately aggressive;
  full usefulness threshold is 500ms. Six roots. Wide bounds or unconfirmed improvements are legitimate outcomes.
- All callback calls, including correctness and warmups, consume shared ledger units. No duplicate engine
  metering, descriptor remeasurement, persistent evaluator cache, proposal calls or paid model requests occur.

Before execution, a preset is defined as acceptable for this workload only if every screen/audit completes,
at least one candidate passes screening, and the audit upper useful-among-rejects proportion is at most .1.
This is a test of declared presets, not a universal production default. Failed presets and uncertainty remain
in the report; do not retune and relabel the same roots as independent validation. Simultaneous empirical coverage
across all study rows is not claimed. The per-challenge family and each audit have their documented separate allocation.

## Run

```powershell
dotnet build benchmarks/EvolutionNoise/EvolutionNoise.csproj -c Release -p:UseLocalEvolution=true -p:EvolutionProjectPath=C:/path/to/Evolution/src/AiDotNet.Evolution/AiDotNet.Evolution.csproj -m:2
dotnet benchmarks/EvolutionNoise/bin/Release/net10.0/EvolutionNoise.dll > noise-study.json
./benchmarks/EvolutionNoise/Verify-Study.ps1 -Report noise-study.json
```

The JSON contains all raw fitness draws/timing observations, failed/incomplete reports, selected reject identities,
policy hashes, sample contexts, ledger totals and hashes of actual consumer/core/tensor assemblies.
Exit success means measurement/accounting contracts passed; inspect `PresetApproved` and confidence bounds for
the actual preset decision. No benchmark speedup is required to keep a valid inconclusive result.
The separate verifier gates the predefined conservative ridge preset and harmful negative controls, checks every
fixed root is present, rejects duplicate within-workflow sample identities and reconciles physical calls against receipts.

## Consumer use and safety

Construct `ProgramNoiseEvaluationOptions` before work; inject correctness, cheap, full, hidden correctness and
hidden full `IProgramFitnessEvaluator` backends into `ProgramNoiseEvaluationSession`. Call `ScreenAndAuditAsync`
on a frozen candidate set with a separately chosen audit seed. It automatically audits complete screen rejects.
`ChallengeAsync` reruns full fitness for both candidate and incumbent; passing a cheap screen alone never promotes.
Check `IsComplete`, audit uncertainty and `IsConfirmed`, retain reports outside the proposer, and recheck incumbent
identity/correctness/applicability before any external archive replacement.

The session pins backend identities and rejects measurement-origin metadata rather than accepting cached or pooled
samples. Raw backends must truthfully perform fresh work; the library cannot detect an unmarked hidden cache.
Correctness is reevaluated for each sample and its cost is included. Backends own actual reset and execution limits.
This study's trusted finite loops and fresh model instances avoid the arbitrary-code isolation requirement; they
do not establish a sandbox for other consumers. Deadlines remain cooperative. CPU scheduling/cache/drift makes
timing independence an assumption, so no production timing superiority is inferred from these six local roots.

Consumer facade/persistent-evaluator regressions plus dedicated tests cover gate failure, restored one-use batches,
cancellation, unknown receipts, budget-short audits and hidden-confirmation rejection. Published-package compatibility
requires the dependency release; source integration against pinned PR #49 is the current verification route.
