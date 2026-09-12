# PR #2154: detection metric and geometry boundaries

This batch addresses review comments `3994110796`, `3994110814`, and
`3994110819`. It changes the shared metric implementation and shared text-test
invariant, not generated leaf tests or detector numerical forwards.

## Contracts and regression controls

Single-threshold AP, mean AP, and precision/recall now reject non-finite IoU
thresholds and thresholds outside `[0,1]`, including inputs with no ground-truth
classes. Null outer lists and mismatched image counts retain their earlier
exception precedence. Invalid thresholds must be rejected before enumerating
either inner detection list; finite endpoints 0 and 1 remain valid. As before,
zero-overlap boxes do not become matches merely because the threshold is zero.

The range API replaces the fixed `1e-9` quotient bias with a correction scaled
to binary64 roundoff. An endpoint mathematically on the grid can be recovered
from division roundoff. If reconstructing that final grid point slightly exceeds
the caller's maximum, it is clamped only within the corresponding arithmetic
tolerance. An off-grid maximum is not appended. The existing 32-threshold batches,
independent match claims, stable ranking, lazy IoU access, and ordered averaging
are unchanged.

The grid controls distinguish `max=0.9999999995, step=1` from a genuine endpoint,
cover an interior near-grid maximum, decimal `0.1..0.3` by `0.1`, the COCO grid,
an off-grid maximum, and a single threshold with `double.Epsilon` step. Existing
controls also require overflow rejection before enumeration for both
`step=1/Int32.MaxValue` and `step=double.Epsilon` over `[0,1]`, where the raw
quotient is infinity. A representable `Int32.MaxValue` threshold count with no
classes still takes the empty result path without allocating matching state.

The shared random-input text invariant now checks all four coordinates for both
NaN and infinity before its unchanged positive-width/height assertions. Four
coordinates times three non-finite values are tested directly. Finite negative
coordinates remain legal for an unclipped EAST box; no image-boundary condition
was added to that contract. The 18 existing generated positive text cases are
included unchanged.

## Failure-before evidence

The first run used the actual production core at
`d8183f1d0e8e8f552b828f33efc459f64318d1ab`, with SHA-256
`69BCCF0B1B5AD45EB4050C3D83D352980BB7F0F8EC4D3153EBEBB5469A4B9B8D`.
For direct testing, the old text helper's visibility was widened from private to
internal, but its assertion body was unchanged. The same 172 cases produced
120 passes, 52 failures, and zero skips:

- 45 invalid single-threshold cases failed their required argument-validation
  contract across three APIs and populated/empty inputs.
- Three range cases failed: the two near-grid maxima admitted a forbidden extra
  threshold, and decimal `0.1..0.3` reconstructed a final threshold above `0.3`.
- Four text cases exposed accepted infinities: negative infinity in left/top,
  positive infinity in right/bottom. Other non-finite combinations were already
  rejected by the old NaN or positive-area conditions.

The other 120 controls passed. The baseline report is
`artifacts/pr2154-review/pr2154-boundaries-first-before.trx`; its test assembly
hash is `B52FB918A966E1A11C9D367CF09E428D96CC6363DB81A2513B0E484AF2963AD1`.
This is actual-library CPU evidence, not a stub or a copied metric implementation.

## Corrected-code results

The 172 cases comprise 15 existing metric cases, 63 existing range/cache cases,
18 generated positive-text cases, 62 new threshold-boundary cases, and 14 new
shared text-geometry cases. No test or assertion was removed to obtain green.

| Actual run | Passed | Failed | Skipped | Report in `artifacts/pr2154-review` |
| --- | ---: | ---: | ---: | --- |
| Historical core and historical text assertions, .NET 10 | 120 | 52 | 0 | `pr2154-boundaries-first-before.trx` |
| Historical core, corrected text assertions only, .NET 10 | 124 | 48 | 0 | `pr2154-boundaries-text-only-before-core.trx` |
| Corrected core and text assertions, .NET 10 | 172 | 0 | 0 | `pr2154-boundaries-final-net10.trx` |
| Corrected core and text assertions, .NET 8 | 172 | 0 | 0 | `pr2154-boundaries-final-net8.trx` |
| Corrected core and text assertions, .NET Framework 4.7.1 | 172 | 0 | 0 | `pr2154-boundaries-final-net471.trx` |
| Independent parent-agent replay, .NET 10 | 172 | 0 | 0 | `pr2154-boundaries-root-independent.trx` |

All three actual core builds succeeded: net10/net8/net471 had zero errors and
2,775/2,775/2,777 warnings respectively, taking 3m40s/4m12s/3m40s. All three
focused test builds had zero errors and zero warnings. The final test executions
took 14/14/20 seconds. The independent net10 replay checked the exact core/test
hashes and passed in 16 seconds.

| Artifact | SHA-256 |
| --- | --- |
| Corrected .NET 10 core | `BB9F13F390C9EB9056891A0BE91E17312851F1095E99F95B512976BF04565E8F` |
| Corrected .NET 10 test assembly | `C230662175B48E3295FD814E0CA59988666FB779050B0BE3D8237D54A9AEB15A` |
| Corrected .NET 8 core | `1EC7D7A5409FD0360239D5FAE3B54B8EAED3251CAA552E23EC6C4EA5BAEAC9AE` |
| Corrected .NET 8 test assembly | `05B40C3CEED00287AEB826B7C0402DE04241B7FDB391816128D8C2D2BB7810CE` |
| Corrected .NET Framework 4.7.1 core | `4FAED56C5CB86ED460CCAB3371918600F91E2ACCB973E45CE851FE47C5A43831` |
| Corrected .NET Framework 4.7.1 test assembly | `F4D7E6337E4D976356376BD7A34CCD3A429B3D7886D7FAFEA10A417E32567D82` |
| Unchanged actual generator | `1AF4448E70ED82A2248EDC7077225B7925ED9F69E78CE642331D71BC5587141F` |

## Focused runner integration

The preceding object-positive fixture added a helper referenced by
`ObjectDetectionTestBase`. Two older focused projects linked the base explicitly
but omitted that helper. Both failed with the same two `CS0103` diagnostics.
The source includes are corrected in `Pr2154.ComputerVision` and
`Pr2154.DetectionParameters`; all six project/framework compile configurations
then succeeded. A repository-wide scan of project/props/targets source files,
including paths outside `review-tests`, found only these two omissions. The
other matching runners already included their required helpers.

These checks used `dotnet msbuild -t:Compile`, so they compiled the current source
into `obj` without replacing the historical test binaries in `bin`. SHA-256
checks confirmed all six frozen binaries were unchanged. Logs are named
`artifacts/pr2154-review/pr2154-focused-links-<project>-<framework>.log`.
The six test hashes and all three earlier positive-object core hashes were
checked again after the compatibility builds and remained unchanged.
This source-list integration check is not a claim that the full main test
project or every model family was executed.

```powershell
foreach ($project in @('ComputerVision', 'DetectionParameters')) {
    foreach ($framework in @('net10.0', 'net8.0', 'net471')) {
        dotnet msbuild "review-tests/Pr2154.$project/Pr2154.$project.csproj" -t:Compile -p:TargetFramework=$framework -p:Configuration=Release -p:BuildProjectReferences=false -p:CopyLocalRuntimeTargetAssets=false -p:CopyLocalLockFileAssemblies=false -p:_GetChildProjectCopyToOutputDirectoryItems=false -m:1 -nodeReuse:false -nologo -v:quiet -clp:ErrorsOnly
        if ($LASTEXITCODE -ne 0) { throw "Focused source compilation failed for $project / $framework." }
    }
}
```

## Reproduction

Run from the repository root. The small runner includes the actual module
initializer, license helper, trace helper, and xUnit configuration. It reuses
actual built core assemblies and the existing CPU native closure; all-RID asset
copying and child Content propagation are disabled. On .NET Framework it uses
the existing managed-only dependency closure target.

```powershell
$framework = 'net10.0' # Also exercise net8.0 and net471.
dotnet restore review-tests/Pr2154.DetectionBoundaries/Pr2154.DetectionBoundaries.csproj -p:NuGetAudit=false
dotnet build src/AiDotNet.Generators/AiDotNet.Generators.csproj -c Release --no-restore -m:1 -nodeReuse:false
dotnet build src/AiDotNet.csproj -f $framework -c Release --no-restore -m:1 -nodeReuse:false -p:BuildProjectReferences=false -p:GeneratePackageOnBuild=false -p:CopyLocalRuntimeTargetAssets=false -p:CopyLocalLockFileAssemblies=false -p:_GetChildProjectCopyToOutputDirectoryItems=false
dotnet build review-tests/Pr2154.DetectionBoundaries/Pr2154.DetectionBoundaries.csproj -f $framework -c Release --no-restore -m:1 -nodeReuse:false -p:BuildProjectReferences=false -p:GeneratePackageOnBuild=false
$cpuNativeDirectory = (Resolve-Path "review-tests/Pr2154.DetectionParameters/bin/Release/$framework").Path
$env:PATH = $cpuNativeDirectory + ';' + $env:PATH
dotnet test review-tests/Pr2154.DetectionBoundaries/Pr2154.DetectionBoundaries.csproj -f $framework -c Release --no-build --no-restore --logger 'trx;LogFileName=boundaries-replay.trx' --results-directory artifacts/pr2154-review
```

The semantic detector-training review remains separate open work. These
boundary checks are not GPU proof, trained detection accuracy, or full-repository
CI proof, and do not by themselves make this draft PR ready to merge.
