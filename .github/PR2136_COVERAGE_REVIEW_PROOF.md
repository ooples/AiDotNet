# Coverage-review regression proof

This evidence covers the two coverage-report review findings, not the separate
MatchaTTS and MGIE architecture findings or a full model-family test run.
The production generator is unchanged by these review fixes.

## Root cause and regression

The synthetic fixture originally had metadata attributes but did not implement
the generic full-model interface required by discovery. Its census was empty.
Checking only measurability therefore could not detect false test attribution.

The fixture now implements that interface with scalar I/O and no matching test
family. Assertions require exactly two models, zero tested, two untested, and an
actually empty generated `TestedModelNames` array. The count assertions include
the terminating semicolon so a larger numeric prefix cannot satisfy them.
Input-compilation errors also fail the test.

Independent Windows/.NET 10 runs used the same valid two-model fixture:

| Assertions | Generator | Result |
| --- | --- | --- |
| Original weak assertions | Isolated old-attribution mutant | 3 passed |
| Corrected strict assertions | Same mutant | 2 passed, 1 failed at `TestedCount = 0;` |
| Corrected strict assertions | Unchanged real generator | 3 passed, no skips |

The mutant differs only by removing the model/self-test exclusion and restoring
the old terminal-name match to `return true`. It lives outside the repository;
the shipped generator was never replaced with the mutant. The corrected tests
were repeated after integrating the newer collaborator commits.

## Reproduce the focused check

From the repository root:

```powershell
dotnet test tools/CoverageReportReview/CoverageReportReview.csproj -c Release --logger "trx;LogFileName=coverage-report-review.trx"
```

This small project references the actual generator project and links the actual
three checked-in generator-driver tests. It does not copy their implementation,
compile the main library, or pretend to rerun the full coverage census. The same
tests remain part of the normal `AiDotNetTests.csproj` test assembly.

For controlled mutation verification, `GeneratorProjectPath` can point to an
isolated copy of the generator project with only the two attribution changes
above. That invocation must fail the strict zero-tested assertion.

## Baseline provenance and open work

The 1485/1816 baseline is a recorded historical measurement, not a new census.
Commit `e9d417a583444fc52c33fcb1c495cf1404d73443` on 2026-09-09 records the
1482-to-1485 change after the structural TimeSeries routing guard. Its provenance
is now documented beside the constant; earlier whole-word reference counts are
not presented as the same measurement.

The remaining MatchaTTS duration/alignment and MGIE image/instruction-conditioning
findings require actual architecture and training-contract work. These focused
test and documentation changes do not resolve them or make this draft merge-ready.
