# Correctness-gated program evolution

Roadmap slice: US-03, with an evidence-preservation prerequisite for US-17/18.

```csharp
var gated = new CorrectnessGatedProgramFitnessEvaluator(publicReferenceChecks, runtimeEvaluator);
var task = new ProgramEvolutionTask(gated, descriptors);
```

`publicReferenceChecks` implements `IProgramFitnessEvaluator` and reports a maximization pass fraction in [0, 1].
Only exactly 1 with zero constraint violations permits runtime evaluation. Report every required check, rather than
averaging only successful cases. A failed, timed-out or rejected check never reaches the runtime evaluator. Positive
fitness-stage constraints also reject the result. Return failures with truthful costs; throwing after spending work
cannot provide an exact receipt to the wrapper.

Given a fast but incorrect program, when reference checks fail, then runtime scoring is not invoked and the program
cannot enter the archive as a completed result.

Given a valid program, when checking and scoring finish, then their same-unit costs are added and scoring metadata
is retained. Correctness-stage success metadata is deliberately not merged into fitness metadata.

Given descriptors are enabled, when ProgramEvolutionTask merges them, then metrics and repair artifacts survive.
That preservation affects future search feedback: task version v3 intentionally refuses old v2 checkpoints.

This is a gate, not a compiler, sandbox or proof of correctness. Both stages must isolate untrusted code and enforce
timeouts. Keep final held-out checks outside search and never expose their cases or diagnostics to the proposing model.
The existing package migration PR #2092 must be reconciled separately; this PR does not modify the copied core engine.
