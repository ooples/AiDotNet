# Compiler-guided C# evolution

Roadmap slices: US-17 syntax edits/compiler repair, US-05 shared resource accounting. Opt-in package:
`AiDotNet.Evolution.CSharp` (net8.0 and net10.0). Compiler dependencies are not added to the main AiDotNet package.
This feature is not a claim that US-17, the full roadmap, or competitor superiority is complete.

## Facade configuration

Use `ConfigureCSharpProgramEvolution(chatClient, programOptions, compilerOptions, resourceOptions)` on the existing
`AiModelBuilder`. The caller supplies an `IChatClient<T>`, trusted reference assemblies, an isolated execution
engine, search-visible fitness and independent correctness checks. No provider, API key or paid service is selected.

```csharp
var compiler = new CSharpProgramEvolutionOptions
{
    ReferencePaths = pinnedReferenceAssemblyPaths,
    TargetIdentity = referencePackAndRuntimeIdentity,
    ModelVersionIdentity = pinnedModelAndProviderConfiguration,
    AuditDirectory = freshEvidenceDirectory,
    MaxRepairs = 2
};
var ledger = new EvolutionResourceLedger(runId, EvolutionResources.Of("cost_units", 100));
var accounting = new ProgramEvolutionResourceOptions(ledger,
    maximumEvaluationCostUnits: 3, compiler.CostUnitVersionHash);

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureProgramExecutionEngine(isolatedExecutionEngine)
    .ConfigureProgramCorrectness(searchVisibleCorrectnessChecks)
    .ConfigureCSharpProgramEvolution(chatClient, programOptions, compiler, accounting)
    .BuildAsync();
```

These identifiers are application-supplied values, not automatically discovered pins. `programOptions` needs seed
source and fitness cases/script as usual. `MaxProposals` includes seeds. The reference paths should name a coherent,
pinned reference bundle for the actual execution target; a compiler cannot prove the caller chose the right target.
Cost limits in this snippet are illustrative synthetic units, not dollars or measured CPU time.

## Acceptance contracts

As an algorithm author, I want edits tied to the exact parent so stale or ambiguous patches cannot change other code.

Given bounded valid C# seed syntax, when proposing a change, then the model receives a bounded catalog of method-body
statements/expressions with original UTF-16 spans, syntax kinds and exact-source SHA-256 hashes. A response must name
the exact parent and non-overlapping catalog nodes, provide a testable hypothesis, and contain exactly the schema's
properties. Duplicate keys, stale spans/hashes, directives, skipped syntax and unchanged candidates are rejected.
Multiple edit spans are applied to the same original snapshot, not sequentially shifted offsets.

Given protected evolve blocks, when applying any syntax patch, then every character outside those bodies and all
marker lines must remain unchanged, including mixed line endings. This protects boundaries, not behavior.

As an algorithm author, I want bounded compiler feedback before spending on runtime fitness.

Given a syntactically valid patch, when validating it, then Roslyn must successfully emit a deterministic Release
C# 12 library with unsafe code disabled and the owned reference images. Parse-only success is insufficient:
[Roslyn documents that compilation diagnostics omit emit-only failures](https://learn.microsoft.com/en-us/dotnet/api/microsoft.codeanalysis.csharp.csharpcompilation?view=roslyn-dotnet-5.0.0).
The package does not load or execute the generated assembly. Failed builds report at most eight diagnostic IDs and
physical source spans; source-containing diagnostic messages and mapped `#line` paths are not returned as feedback.

Given a failed build or invalid patch, when repairs remain, then the next request retains the original parent/catalog
and only the latest failed reply plus bounded feedback. Exhaustion cannot produce an uncompiled child. A successfully
compiled child still passes the configured correctness gate and normal fitness pipeline; compilation is not acceptance.

As an experiment owner, I want failed work and unknown consumption included in the same resource ledger.

Given a shared ledger, when a proposal starts, then its complete worst-case model/parse/build/audit reservation must
fit before dispatch. Reference loading is charged once as setup. Actual failed/successful attempts reconcile model
calls, reported tokens, syntax attempts, dispatched emits and committed evidence. Missing usage or a provider/audit
exception retains the reserved maximum as unknown consumption. Reported overruns are never clipped to the reservation.
Per-call overages reject the proposal; an aggregate resource-maximum violation additionally closes ledger admission.

Given program evaluation, when correctness checks and fitness finish, then their returned same-unit costs are charged
once through the shared ledger. The built-in I/O evaluator counts dispatched runner calls, including canceled calls;
it does not measure CPU time. Custom evaluators must supply truthful same-unit costs. Optional ledger dimensions are
`model_calls`, `input_tokens`, `output_tokens`, `parse_calls`, `build_calls`, `audit_calls`, `artifact_bytes` and
`reference_bytes`. Undeclared dimensions are not enforced. Prompt-byte admission is conservative and does not claim
knowledge of a provider's hidden framing/tokenizer; provider-reported usage is still required.

As a reviewer, I want evidence identifying what was proposed, compiled and paid for.

Given a completed attempt, when accepting its child, then a write-once evidence record must first be committed beneath
the caller's trusted directory. Records include exact parent/proposed source, prompts, retained response, hypothesis,
feedback, compiler outcome/image hash, reference hashes, assembly dependency fingerprint, declared target/model/cost
identities, bounds/prices and cumulative attempt costs. Responses beyond the configured bound are explicitly truncated;
base64 preserves the exact retained UTF-16 code units, including malformed Unicode. A truncated response is not a
complete replay transcript. A file collision or failed write fails closed; no existing evidence is overwritten.
Unlike the legacy redacted provenance sink, this exact-source evidence contains unredacted source, prompts and
model output. Treat it as sensitive: use a private access-controlled directory, define retention/encryption, and
never supply credentials or sealed test cases as search inputs. No automatic upload is performed by the package.

## Limits and operational requirements

- Reference images are owned copies: at most 64 files, 32 MiB each, 128 MiB total. Candidate source is at most 65,536
  UTF-16 characters; emitted PE is bounded to 8 MiB; each audit record is bounded to 2 MiB. These are not peak-RAM limits.
- Compiler cancellation is cooperative (1–30 seconds), **not an OS security sandbox or hard CPU/memory timeout**.
  Isolate the host appropriately for hostile inputs; execute candidates only through the caller's isolated runner.
- No correctness-test-driven repair, cross-run experience retrieval, dependency rewriting, multi-file project edits,
  durable distributed workers, or actual held-out performance promotion is supplied by this slice.
- Final sealed tests and their feedback must remain outside search. Check public API behavior and dependencies against
  the execution target independently. Source-span restriction alone does not prove behavior or public API preservation.
- Options snapshot configuration, but the facade deliberately shares the live operator/ledger. Use fresh instances and
  evidence directories for independent runs. Automatic engine checkpoint/resume is refused until its transaction can
  also persist the ledger; standalone operator state is not a complete resumable experiment.
- This branch requires the new AiDotNet.Evolution ledger/proposal APIs. The older published `0.1.0-preview.1` does not
  contain them. Use `UseLocalEvolution=true` and the matching source checkout for development; release requires a new
  compatible core package, a dependency pin and a clean PackageReference consumer test. No package publication is claimed.

## Verification

`tests/AiDotNet.Evolution.CSharp.Tests` invokes real Roslyn parsing/emit using local reference images and scripted
chat replies; it tests protected edits, failed compilation/repair, evidence and shared-ledger facade wiring.
The execution double tests wiring only; it is not a runtime benchmark. No live model or paid API call is used.
Local .NET 8/10 verification passed all 62 compiler-package tests and all 876 focused consumer tests. The compiler
package's .NET 10 run measured 499/505 covered lines (98.81%) and 331/360 branches (91.94%). Source/test DLL hashes
were compared before execution. These numbers do not cover net471, default NuGet dependency resolution, full CI,
live models or representative runtime benchmarks. TRX and coverage were retained under `TestResults/compiler-verified-*`
and `TestResults/complete-receipts-*`; these generated local artifacts are not committed.
The dedicated workflow checks a pinned companion source revision and retains TRX/coverage; it does not replace normal
package-path release validation. See [implementation status](https://github.com/ooples/AiDotNet.Evolution/blob/feat/competitive-evolution-platform/docs/IMPLEMENTATION_STATUS.md)
for the remaining roadmap acceptance work.
