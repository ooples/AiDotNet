# Sequence scaffold review proof (#2128)

The same 13 contracts run the real source generator with synthetic compiler inputs.
At PR head `671836a347436f4b023ebe3f02a86ea333ad0686`, four structural checks fail
and nine controls pass. After removing the unreachable XLSTM/Hawk rules and the
shadowed Griffin/Hawk fallback arms, all 13 pass, with no skips.

All six effective-fixture checks pass **before and after**: XLSTM, Griffin, Hawk,
GLA, GatedDeltaNet and RecurrentGemma retain their bounded dimensions. This change
does not select the previously unreachable XLSTM dimensions or change production
model defaults, seeding, test tolerances, timeouts or iteration budgets.

Run from the repository root:

```powershell
dotnet test tools/SequenceFixtureReview/SequenceFixtureReview.csproj -c Release --logger 'trx;LogFileName=sequence-generator-after.trx' --results-directory artifacts/sequence-generator
```

For a controlled original-head comparison, materialize that exact Git revision in
a separate checkout, then point `GeneratorProjectPath` and `GeneratorSourcePath`
at its generator project and `TestScaffoldGenerator.cs`. Both properties resolve
relative to this review project unless absolute paths are provided. Keep the
current contract test file unchanged. Run before/after sequentially because this
project shares its output directory between arms.

Recorded local reports: `artifacts/sequence-generator/sequence-generator-before.trx`
(4 failed, 9 passed) and `sequence-generator-after-confirmed.trx` (13 passed).
The baseline source was extracted directly with `git archive` from the head above;
it was not reconstructed by selectively undoing fixes.

These tests verify emitted syntax and unique constructor selection. Their model
symbols are synthetic: they do **not** prove actual model construction, training,
GPU execution, semantic compilation against every real constructor, or a passing
CI shard. The separate real-model/options tests supply their own evidence. The
contracts are also included in the normal test project; no generated leaf test
was edited by hand.
