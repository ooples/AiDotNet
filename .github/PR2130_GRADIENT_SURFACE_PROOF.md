# PR #2130: remove the unconsumed gradient alias

Baseline: `faed3685ac40d5a074de7375f2ab400120e095c2`.
Review: [3964560779](https://github.com/ooples/AiDotNet/pull/2130#discussion_r3964560779),
thread `PRRT_kwDOKSXUF86ggA2f`.

## Root cause and compatibility boundary

`ModelHyperparameterOptions.MaxGradNorm` was introduced with that new base in
`9f7fe07b6ce05bb282281f0ea59b78d72f5f9ded` (options migration phase 1). The file
did not exist at that commit's parent, and the preceding public base classes
`NeuralNetworkOptions` and `ModelOptions` had no gradient-limit property.

The migration did not connect this new property to a model or optimizer.
Removing its declaration, initialization, and copy assignment removes a no-op
setting without installing a second clipping policy. The stale review example
about copying was already fixed by the shared copy constructor; copying an
inert alias correctly did not make it operational.

The existing `MaxGradientNorm` and `EnableGradientClipping` properties on
Finch/Griffin/Hawk/RecurrentGemma remain unchanged. This change does not claim
to implement any independently missing Finch recipe behavior.
The real `NeuralNetworkBase.MaxGradNorm`, `MaxGradNormValue`, eager/compiled
clipping code, optimizer configuration, and independent PPO/GraFPrint options
are untouched. In particular, `GraFPrintOptions` derives directly from
`ModelOptions` and declares its own `MaxGradNorm`; it is not the removed alias.

## Executed evidence

| Run | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Six focused controls before production removal, net10.0 | 4 | 2 | 0 |
| Full scalar project after removal, net10.0 | 395 | 0 | 0 |
| Full scalar project after removal, net8.0 | 395 | 0 | 0 |
| Full scalar project after removal, net471 | 395 | 0 | 0 |

Before removal, the two intended failures detect the public alias and its
emitted XML promise. The four positive controls already pass and continue
passing: each existing optimizer threshold defaults to 1.0, its enable flag
defaults to true, and copies retain independent non-default threshold/flag
values. The exhaustive inherited-copy and existing optimizer validation tests
also pass unchanged.

The previous documentation test promising behavior for the unused property
is replaced by the assertion that the XML no longer advertises that property.
The obsolete default-value assertion for that removed property is dropped;
all other sequence default values and the 17+1 type census are unchanged.
Five shared API/optimizer tests are added, increasing the prior 390 tests to 395.

Artifacts: `artifacts/pr2130-gradient-surface/results/gradient-surface-baseline.trx`
and `gradient-surface-final-<tfm>.trx` in the same directory.

Final DLLs live in
`artifacts/pr2130-gradient-surface/build/bin/AiDotNet.OptionsContractTests/release_<tfm>/`:

| TFM | `AiDotNet.OptionsContractTests.dll` SHA-256 |
| --- | --- |
| net10.0 | `4C719C76575E14787F964FC6925750081FA98E5F8D4BAA4D60F76731C12D5D8D` |
| net8.0 | `80425CF907A290A3744296D846B83A35AC43BA039580E7F6CB100706C0872262` |
| net471 | `136548BBF5E79E3943FB80471D245A8CAD394E1A3F57D90914C2AC96B482B5F3` |

Reproduce from the repository root:

```powershell
foreach ($gradientTfm in @('net10.0', 'net8.0', 'net471')) {
    dotnet test tests/AiDotNet.OptionsContractTests/AiDotNet.OptionsContractTests.csproj `
        -c Release -f $gradientTfm `
        --artifacts-path artifacts/pr2130-gradient-surface/build `
        -m:1 -p:UseSharedCompilation=false `
        --logger "trx;LogFileName=gradient-surface-final-$gradientTfm.trx" `
        --results-directory artifacts/pr2130-gradient-surface/results -v:quiet
    if ($LASTEXITCODE -ne 0) { throw "Gradient surface suite failed on $gradientTfm." }
}
```

The recorded final executions used `--no-restore` after the baseline restore.
For the failure-first control, apply only the new shared test file, its project
source link, and the XML test update to the baseline in a separate worktree;
leave production and the original default assertion unchanged. Filter with
`FullyQualifiedName~ModelHyperparameterSurfaceTests|FullyQualifiedName~SharedFamilyDocumentation_DoesNotAdvertiseAnUnusedGradientAlias`.
Require a nonzero test exit and a TRX containing exactly six executed cases,
four passes and the two intended alias/XML failures.

This is an options-boundary correction. No full-library/native-model build or
GPU/eager/compiled-training rerun is claimed by this scalar evidence. Those
production algorithms and earlier proof artifacts were not modified.
