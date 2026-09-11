# Correctness-gated program evolution

Roadmap slice: US-03, with an evidence-preservation prerequisite for US-17/18.

```csharp
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureChatClient(chatClient)
    .ConfigureProgramCorrectness(publicReferenceChecks)
    .ConfigureProgramEvolution(programOptions)
    .BuildAsync();
```

`programOptions` provides the usual fitness test cases or evaluator script. The gate is internal plumbing assembled
by `AiModelBuilder`; no additional public helper needs to be constructed. `publicReferenceChecks` implements
`IProgramFitnessEvaluator` and reports a maximization pass fraction in [0, 1].
Only exactly 1 with zero constraint violations permits runtime evaluation. Report every required check, rather than
averaging only successful cases. A failed, timed-out or rejected check never reaches the runtime evaluator. Positive
fitness-stage constraints also reject the result. Return failures with truthful costs; throwing after spending work
cannot provide an exact receipt to the wrapper.
An absent result is likewise unknown consumption: gate v2 throws instead of claiming that zero or just the
checking-stage cost is the complete total. Shared resource accounting retains the reserved maximum in that case.

Given a fast but incorrect program, when reference checks fail, then runtime scoring is not invoked and the program
cannot enter the archive as a completed result.

Given a valid program, when checking and scoring finish, then their same-unit costs are added and scoring metadata
is retained. Correctness-stage success metadata is deliberately not merged into fitness metadata.

Given descriptors are enabled, when ProgramEvolutionTask merges them, then metrics and repair artifacts survive.
That preservation and exact-source identity affect future search feedback: task version
`program-evolution-task-v5-cost-semantics-and-marker-identity` intentionally refuses older checkpoints. It also
separately fingerprints start/end markers and declared resource cost-unit semantics.

## Breaking change and migration

`IAiModelBuilder<T, TInput, TOutput>` now requires `ConfigureProgramCorrectness(IProgramFitnessEvaluator)`.
External implementations must add this member and recompile. Store or forward the supplied evaluator and ensure
it gates program fitness as described above; silently ignoring it would bypass the requested correctness checks.
Implementations that cannot support program evolution should explicitly throw `NotSupportedException`.
Users of the provided `AiModelBuilder` need no implementation changes. See the unreleased breaking changes in
[CHANGELOG.md](../CHANGELOG.md) and [source-identity migration](evolution-source-identity.md).

This is a gate, not a compiler, sandbox or proof of correctness. Both stages must isolate untrusted code and enforce
timeouts. Keep final held-out checks outside search and never expose their cases or diagnostics to the proposing model.
The #2092 package-migration commits are integrated into this companion branch (not merged into master).
The engine now comes from AiDotNet.Evolution; see [package integration](evolution-package-integration.md).
