# PR #2100 review-fix proof

Baseline: `8c1a059c73254136c076a540ac5b82504e3e5499`.

The whole existing `TradingAgentLearningTests` class was run against real
AiDotNet source, its generators, the repository deterministic CPU initializer,
and published AiDotNet.Tensors 0.130.3. A narrow temporary test project compiled
that class and its normal global imports, not replacement agent implementations.
This is not a full model-family or repository test-run claim.

| Source | Observed result |
| --- | --- |
| Unchanged baseline, complete finance class | 8 passed, 3 failed, 0 skipped |
| Fixed class with direct-update/invalid-update/mixed-replay controls | 11 passed, 0 failed/skipped, repeated |
| Independent final repetition | 11 passed, 0 failed/skipped |

The seeded A2C test fixes and verifies the actual policy parameters, then checks
all 600 one-hot actions against the configured random stream. The old sampler
ignored that stream. Sampling now reuses the seeded base-class random source,
also avoiding per-draw secure-random creation.

Two DQN regressions demonstrate actual online-weight changes while the old
direct-update path left TrainingSteps at zero. Both direct gradients and replay
now complete the same training step: one counter advance, epsilon update and
scheduled target copy. Tests inspect real online and target parameters, prove
staleness between copies, exact equality at copy boundaries, and no mutation or
schedule advance for a rejected gradient. No tolerance was widened.

The documentation thread was already addressed at the baseline: IsFinite,
Softmax and SampleAction each have an attached summary/remarks block.

For a normal repository checkout, the same checked-in cases run with:

```powershell
dotnet test tests/AiDotNet.Tests/AiDotNet.Tests.csproj -c Release -f net10.0 --filter FullyQualifiedName~TradingAgentLearningTests --logger "trx;LogFileName=finance-review.trx"
```

The independent bounded local execution used the temporary source-linked harness
and preserved `before-full-finance.trx`, `after-review-fixes-final.trx`,
`after-review-fixes-final-repeat.trx`, and `after-root-independent.trx`.
Baseline AiDotNet.dll SHA-256:
`CFC71687656A1E6C45B893AEEB149069571E6E20A9EBEAC1513B044D59F55F74`.
Fixed AiDotNet.dll SHA-256:
`63B0EC5F45EC39BB57D98BA6F6885180B33237FD5A9CCD25901E4FA28BFD6913`.
Builds succeeded with existing analyzer warnings. No unpublished dependency,
null-forgiving operator, public test hook, or new string-based policy mode was
introduced.
