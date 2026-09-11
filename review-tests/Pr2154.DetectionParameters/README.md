# PR #2154: live parameter-layout and metadata proof

This batch fixes shared parameter exposure needed by the detector review. It does
**not** claim that positive detection fixtures or task-specific detector training
are complete. The separate AP-cache proof remains in
[`../Pr2154.ComputerVision/AP_CACHE_PROOF.md`](../Pr2154.ComputerVision/AP_CACHE_PROOF.md).

## Root defects and contracts

The five shared backbone adapters exposed actual parameter chunks but not their
underlying layouts. The registry correctly refused to treat an unverified chunk
source as writable and used a detached fallback. `CvParameterModule` and RPN had
the same missing layout contract. They now delegate the real layer layout; the
registry's safety gate and its negative controls are unchanged.

Shared module layout and chunks use the same canonical indexed child IDs. Tests
check tensor identity, role, flat order, normalized offsets, zero-sized own
slots, null children, deferred shapes, and every propagated layout descriptor.
Accessor and collection registration both expose actual tape weights: a real
gradient update must change the real forward and reduce an independently defined
sum objective. No replacement forward or synthetic optimizer is used.

Exposing the real MHA layout uncovered a second defect: querying optional child
structure called its value initializer and allocated projection weights. The
base still initializes unknown child structures. Only a conservative generator
proof can omit this step for an exact runtime layer type; derived types retain
the original path. Nullable fields alone are not evidence that children are
ready, and collections, inherited/unknown structural paths, owner escapes,
callbacks, setters, user-defined operators, and unknown external contracts
retain initialization.

The proof assumes the explicitly resolved numeric, tensor, engine, and
initialization APIs honor their value-only contracts. An initialization strategy
that secretly captures its owner to create child layers violates that contract.
This is not general whole-program side-effect analysis. Metadata tests use a
rejecting initialization strategy to prove that weight initialization is not
invoked at all, not merely that it leaves the same values behind.

## Failure-first evidence

TRXs and logs are retained locally in `artifacts/pr2154-review/`; they are not
committed binaries. Counts below are TRX executed/passed/failed counts, not just
process exit codes. The original adapter baseline is the frozen AP-complete
core (commit `7ccc544f94694dd41c6a73350473f0c397bac023`, core SHA256
`2D5355184C81886DA018076141A8A1030F9E5C49527C8DDB145B75253399388C`).

| Snapshot / control | Evidence file | Passed / executed |
| --- | --- | ---: |
| Original adapter regression cohort | `pr2154-adapter-live-before.trx` | 20 / 50 (30 failed) |
| Expanded original adapter cohort | `pr2154-adapter-live-expanded-before.trx` | 20 / 56 (36 failed) |
| Layout delegation before canonical child-ID and metadata fixes | `pr2154-adapter-live-expanded-after.trx` | 51 / 56 (5 failed) |
| Canonical IDs, before actual MHA initializer eligibility | `pr2154-adapter-live-intermediate.trx` | 57 / 59 (2 failed) |
| Actual MHA eligibility, before final primitive-contract tightening | `pr2154-adapter-cv310-intermediate.trx` | 310 / 310 |
| Adversarial setter / owner-alias controls before correction | `pr2154-metadata-generator-adversarial-red.trx` | 12 / 19 (7 failed) |
| Reassigned-null callback controls before correction | `pr2154-metadata-generator-callback-red.trx` | 20 / 23 (3 failed) |
| Constructor / user-operator controls before correction | `pr2154-metadata-generator-operator-red.trx` | 22 / 32 (10 failed) |
| Implicit callback controls before correction | `pr2154-metadata-generator-implicit-effects-red.trx` | 38 / 41 (3 failed) |
| Non-primitive `SpecialType` controls before correction | `pr2154-metadata-generator-specialtype-red.trx` | 41 / 44 (3 failed) |
| All final policy cases against the exact unchanged baseline generator | `pr2154-metadata-generator-final44-baseline.trx` | 36 / 44 (8 failed) |
| Final 44 generator policy controls | `pr2154-metadata-generator-specialtype-after.trx` | 44 / 44 |

The intermediate real-core SHA256 was
`69BCCF0B1B5AD45EB4050C3D83D352980BB7F0F8EC4D3153EBEBB5469A4B9B8D`.
That earlier runtime result proved actual MHA eligibility but was not the final
generator-policy snapshot. The final reviewed-source rebuild and replays below
produced the same net10 core hash: the additional conservative rejects do not
change emitted code for these actual layer types. The source-level adversarial
controls, not an assumed change to the core DLL hash, prove those rejects.

## Final reviewed-source validation

All three actual core builds completed with zero errors. Build logs are
`pr2154-layout-reviewed-core-net10.log` (2775 warnings, 2m52s),
`pr2154-layout-reviewed-core-net8.log` (2775 warnings, 2m57s), and
`pr2154-layout-reviewed-core-net471.log` (2777 warnings, 2m59s). These are core
compatibility builds plus bounded actual-source test runners, not a claim that
the entire main test assembly or every model-family shard ran.

| Target | Runtime TRX | Passed / executed / skipped |
| --- | --- | ---: |
| net10.0 | `pr2154-layout-reviewed-net10.trx` | 310 / 310 / 0 |
| net8.0 | `pr2154-layout-reviewed-net8.trx` | 310 / 310 / 0 |
| net471 | `pr2154-layout-reviewed-net471.trx` | 310 / 310 / 0 |

The actual core and copied runner DLL hashes agree for each target:

- net10.0: `69BCCF0B1B5AD45EB4050C3D83D352980BB7F0F8EC4D3153EBEBB5469A4B9B8D`
- net8.0: `97F5D10424D8B75698A144E248797B18A149F08B376727361A37E609E82B27BA`
- net471: `B6E19F6F44C9F6E43FFA2A9A29B14B9999B9F76F86669F273644027CA6BEB64B`

### Harness negative controls

The initial isolated generator harness omitted the real test initializer and
failed two existing semantic-compilation tests against both the old and new
generator. With the actual `ModuleInitializer.cs`, licensing helper, and core
dependency graph, the unchanged baseline passes 59/59
(`pr2154-metadata-generator-initialized-baseline.trx`). No semantic assertion was
weakened. That unchanged generator DLL has SHA256
`D71ABAE9452767B828CEFB6379D25A2A78E5F38016DD36F3F11A8A3B64D73CB0`;
the reviewed generator DLL has SHA256
`488763248BE3B050EB1EACB8BAE44CBAECAE80FCC67797C6A436B83D22993A53`.
The final generator then passes all 59 existing plus 44 new cases:

| Target | Evidence file | Passed / executed / skipped |
| --- | --- | ---: |
| net10.0 | `pr2154-generator-reviewed-restored-net10.trx` | 103 / 103 / 0 |
| net8.0 | `pr2154-generator-reviewed-net8.0.trx` | 103 / 103 / 0 |
| net471 | `pr2154-generator-reviewed-net471.trx` | 103 / 103 / 0 |

After the last historical baseline control, the net10 runner was rebuilt with
the reviewed generator; its copied generator hash was checked and all 103 cases
were rerun. No runner is left pointing at the baseline generator.

An earlier net471 launch exited zero but discovered no tests because managed
xUnit dependencies were missing; it is **not** a passing result. The corrected
runner copies the SDK-resolved managed runtime assemblies on .NET Framework,
which cannot use `.runtimeconfig.dev.json` NuGet probing. It does not copy every
platform's native runtime assets. The repository's actual CPU initializer and
xUnit configuration remain in both runners.

## Focused reproduction

Run from the repository root. Restore the two runner projects once. Build the
generator and actual core for the chosen target before using `--no-build` tests;
`BuildProjectReferences=false` deliberately prevents a hidden large rebuild.
The detection runner contains 310 unique cases: the existing 271 CV/AP cases
and 39 new shared-adapter cases. The generator runner contains 103 cases.

```powershell
dotnet restore review-tests/Pr2154.DetectionParameters/Pr2154.DetectionParameters.csproj
dotnet restore review-tests/Pr2154.LayerStructureGenerator/Pr2154.LayerStructureGenerator.csproj
dotnet build src/AiDotNet.Generators/AiDotNet.Generators.csproj -c Release --no-restore -m:2 -nodeReuse:false

$reviewTarget = 'net10.0' # Repeat with net8.0 and net471.
dotnet build src/AiDotNet.csproj -f $reviewTarget -c Release --no-restore -m:2 -nodeReuse:false -p:BuildProjectReferences=false -p:GeneratePackageOnBuild=false -p:CopyLocalLockFileAssemblies=false
dotnet build review-tests/Pr2154.DetectionParameters/Pr2154.DetectionParameters.csproj -f $reviewTarget -c Release --no-restore -m:2 -nodeReuse:false -p:BuildProjectReferences=false -p:GeneratePackageOnBuild=false
dotnet build review-tests/Pr2154.LayerStructureGenerator/Pr2154.LayerStructureGenerator.csproj -f $reviewTarget -c Release --no-restore -m:2 -nodeReuse:false -p:BuildProjectReferences=false -p:GeneratePackageOnBuild=false

dotnet test review-tests/Pr2154.DetectionParameters/Pr2154.DetectionParameters.csproj -f $reviewTarget -c Release --no-build --no-restore --logger "trx;LogFileName=pr2154-layout-$reviewTarget.trx" --results-directory artifacts/pr2154-review
dotnet test review-tests/Pr2154.LayerStructureGenerator/Pr2154.LayerStructureGenerator.csproj -f $reviewTarget -c Release --no-build --no-restore --logger "trx;LogFileName=pr2154-generator-$reviewTarget.trx" --results-directory artifacts/pr2154-review
```

These bounded runners suppress transitive platform-native Content copies. Supply
the ordinary CPU-native dependencies through an existing validated CPU runtime
directory on the native library search path. Do not copy another runner's
managed dependency directory wholesale: the generator runner deliberately uses
its own resolved Roslyn 4.14.0 package version. On Windows, an existing CPU-native
directory can be prepended to `PATH` for this shell. This is CPU correctness
proof, not a physical-GPU performance claim.

Verify the expected executed counts as well as the exit status:

```powershell
$reviewCases = @{
    "pr2154-layout-$reviewTarget.trx" = 310
    "pr2154-generator-$reviewTarget.trx" = 103
}
foreach ($reviewCase in $reviewCases.GetEnumerator()) {
    [xml] $reviewResult = Get-Content -LiteralPath (Join-Path artifacts/pr2154-review $reviewCase.Key)
    $reviewCounters = $reviewResult.TestRun.ResultSummary.Counters
    if ([int]$reviewCounters.total -ne $reviewCase.Value -or
        [int]$reviewCounters.executed -ne $reviewCase.Value -or
        [int]$reviewCounters.passed -ne $reviewCase.Value -or
        [int]$reviewCounters.failed -ne 0 -or
        [int]$reviewCounters.notExecuted -ne 0) {
        throw "Unexpected test result counts: $($reviewCase.Key)"
    }
}
```

## Remaining review scope

Diagnostic runs demonstrate that CRAFT, DBNet, EAST, and all nine object
detectors now expose live trainable chunks. Controlled actual text heads produce
nondegenerate output. These probes are not yet shared positive fixture tests,
and tied object scores do not prove sorting or NMS. Those reviews remain open.
Likewise, preserving raw-output regression training does not implement typed
detection assignment/classification/box losses; that is a separate unfinished
review item.
