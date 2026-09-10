# Non-runtime CI routing canary

This documentation-only file exercises the permanent `ci-proof/**` workflow trigger. Its pull
request must skip build, test, parameter, and model validation. After merge, the exact-tree
certificate must prevent that validation from being repeated on `master` while required quality
checks retain their independently typed execution decision.

## Executable contract

The checked-in [workflow contract](../tools/TestImpact/Test-CiImpactWorkflow.ps1) fails unless
every expensive validation job consumes the typed execution decisions. The
[validation resolver](../tools/TestImpact/Resolve-CiValidationReuse.ps1) validates certificate
schema, exact Git tree, source run, scope, and artifact provenance before emitting those
decisions. Run the complete local contract with:

```powershell
./tools/TestImpact/Test-CiImpactWorkflow.ps1
./tools/TestImpact/Test-ValidationReuseModes.ps1
./tools/TestImpact/Test-CiGateModes.ps1
```

For a non-runtime pull request, the required outputs and job results are:

- `requires_validation=false` and `matrix=[]`;
- Build and compatibility-build jobs skipped with no runner;
- none of the 116 primary test shards instantiated, with the empty matrix placeholder skipped;
- parameter and model-shape matrices skipped with no runner;
- regression analysis, aggregate analysis, and artifact-size checks skipped with no runner;
- Sonar receives the typed non-runtime value and its report step succeeds;
- the validation gate succeeds and publishes a validation-only certificate without waiting for
  CodeQL;
- after merge, exact-tree reuse emits `execute_validation=false` and starts none of those
  validation runners while quality checks follow the independent `execute_quality` decision.

## Recorded live proof

[Pull-request run 34412630421](https://github.com/ooples/AiDotNet/actions/runs/34412630421)
produced `requires_validation=false` and `matrix=[]`. Its 17-job graph skipped Build, compatibility
build, all 116 primary shards, parameter sweeps, model-shape windows, regression analysis,
aggregate analysis, and size checks without assigning runners. Sonar job
[102670503633](https://github.com/ooples/AiDotNet/actions/runs/34412630421/job/102670503633)
completed successfully through `Report non-runtime validation`. Validation gate job
[102670504055](https://github.com/ooples/AiDotNet/actions/runs/34412630421/job/102670504055)
succeeded, and proof job
[102670562875](https://github.com/ooples/AiDotNet/actions/runs/34412630421/job/102670562875)
published artifact `ci-expensive-validation-certificate-a7642e20e0523d6fea326c54a10d47c0c746c530`
as schema 3, scope `Validation`, `requiresValidation=false`, and tested tree
`b478ae06bd9f0646f482a48ce73bcbd597a32c6e`.

[Master run 34412846721](https://github.com/ooples/AiDotNet/actions/runs/34412846721) has the
same tree `b478ae06bd9f0646f482a48ce73bcbd597a32c6e`. Resolver job
[102670992956](https://github.com/ooples/AiDotNet/actions/runs/34412846721/job/102670992956)
emitted `reuse_scope=Validation`, `reused_requires_validation=false`,
`execute_validation=false`, and `execute_quality=true`. Its build, compatibility, 116-shard,
parameter, model-shape, regression, aggregate, size, and validation-gate jobs all remained
runnerless and skipped. Sonar's non-runtime report succeeded; CodeQL reran as required.
