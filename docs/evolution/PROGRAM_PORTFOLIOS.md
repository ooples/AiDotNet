# Program proposal portfolios (US-08)

`ProgramVariationPortfolio` plugs into `ProgramEvolutionOptions.CustomVariation` without changing the evaluator, compiler worker or program facade. It is opt-in; no default strategy changes.

Build metered mutation, crossover, restart, local-refinement and consumer-provided model arms using `MeteredProgramVariationOperator`. Each backend implements `ICostedProgramProposalSource`: return an explicit receipt for **all** work inside the proposal, including refinement, model attempts, parsing, compilation, repair and audit writes. Failures stay charged; missing consumption reserves the declared maximum rather than granting free work. Return cumulative usage through `GetUsage()`. Do not meter the same backend work twice.

Existing compiler-guided model arms can be constructed with `CSharpProgramEvolutionExtensions.CreateCSharpProgramVariation(client, program, compilerOptions, resources)`. Construction validates reference metadata and immediately charges setup, but makes no model call. Give each arm a distinct compiler ID and audit directory. The original `ConfigureCSharpProgramEvolution` path reuses this factory.

```csharp
var portfolio = new ProgramVariationPortfolio(arms,
    new EvolutionOperatorRewardPolicy(
        EvolutionOperatorRewardKind.ParentImprovement,
        EvolutionOperatorCostBasis.ProposalAndEvaluation,
        resources.CostUnitVersionHash), explorationProbability: 0.2);
program.ResourceAccounting = resources;
program.CustomVariation = portfolio;
portfolio.CreditCommitted += credit => WriteIdempotentAudit(runId, credit);
builder.ConfigureProgramEvolution(program);
```

`arms`, `resources`, `program`, `builder` and the audit sink are caller-owned. Every arm must implement `IProgramResourceLedgerProvider` and use **the same live ledger instance** as the evaluator; equal unit strings on different ledgers are insufficient. The built-in metered adapter provides that binding. Unmetered plain LLM operators are rejected instead of pretending provider calls were free. Resource limits/conversions are explicit caller settings, not inferred prices or subscription entitlements.

Credit includes child/configuration identity, evaluation/generation identity, archive outcome, attempts, proposal receipt, evaluator cost and bounded applied reward. In-process notification handlers are invoked once per committed outcome, exceptions are counted and isolated, and reentrant mutation/checkpointing is rejected. This is not a transactional exactly-once external delivery guarantee: the sink should use run + evaluation identity and make failures visible. CaptureState preserves learned totals, pending attribution and child state, not notification delivery history.

Restore a fresh equivalent portfolio and the matching shared ledger at the same boundary. Child models/prompts/compiler configuration and reward units must match; changing those invalidates state. Randomness comes from the engine's restored streams. Automatic facade resume with live resource accounting remains explicitly refused until the caller can coordinate ledger and engine checkpoints; this adapter does not silently waive that restriction.

Compiler integration tests use a scripted provider to exercise real parsing, emitting, failed repair costs and facade feedback without paid calls. They are correctness/accounting evidence, not model-quality benchmarks. Core PR #51 separately compares adaptation against development-selected static operators on different held-out local tasks. Broader model-driven or production-speedup claims require their own evidence.
