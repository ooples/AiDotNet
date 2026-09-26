# PR #2154: generated positive text-detector proof

This batch addresses review comment `3985472507`. It adds one shared positive
invariant and a typed factory emitted by the actual test scaffold generator for
CRAFT, DBNet, and EAST. No generated leaf test is edited by hand. Production
models, forward methods, decoding, and thresholds are unchanged.

## What is proved

The shared fixture draws a deterministic 64-by-64 `HI` bitmap without a font or
external image dependency. A separate generated factory supplies the bounded
Nano profile; ordinary generated factories retain their original defaults.
After a real forward initializes lazy shapes, every trainable chunk must be
writable in place. The fixture controls those actual weights and calls the
production `Predict` and `Detect` paths, without replacing `Forward`.

| Detector | Controlled positive result at threshold 0.05 |
| --- | --- |
| CRAFT | One region, confidence 0.5, box `(0, 0, 60, 60)` |
| DBNet | One region, confidence 0.5, box `(0, 0, 63, 63)` |
| EAST | 64 eligible score cells and known RBOX distances; real IoU-0.2 suppression leaves eight regions, first box `(-20, -12, 28, 20)` |
| EAST, separated geometry | The same live head changed to distances 0.25 produces 64 nonoverlapping 4-by-4 boxes at exact grid coordinates |

All positive regions must have a nonempty finite polygon enclosing area, a
positive finite box containing that polygon, the expected confidence, and the
correct source-image dimensions. The EAST raw-output assertions cover all 384
values of the documented flattened `[1, 384]` public `Predict` result: 64 scores
and five geometry channels. With the same model and image, raising the detection
threshold to 0.75 must reject every known 0.5-confidence region.

The exact shared geometry assertions also run through the original random-input
factories. Their existing empty-result and box-only behavior is preserved; the
new positive invariant separately rejects empty results and empty polygons.

This is a **controlled numerical-pipeline and decoder proof**, not evidence of
trained text-recognition accuracy or image-to-label generalization. Zeroing the
weights intentionally makes the known scores analytically predictable. It is
CPU validation, not a physical-GPU or full-repository shard result. The distinct
object-detector positive-fixture and semantic-training findings remain open.

## Actual before/after evidence

The focused suite has 18 cases: three generated-factory contracts, three real
generated positive fixtures, three unchanged default-profile geometry replays,
one valid-result oracle control, and eight malformed/empty-result controls.
Invalid controls cover empty regions, empty/degenerate/nonfinite polygons,
inverted/nonfinite boxes, invalid confidence, and a box not containing its
polygon. The nonfinite-box and containment controls adjust the separately
expected first box so another exact-box comparison cannot mask the relevant
missing guard.

| Run | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Final 18-case suite, prior generator from `1d8961633155aefd881d7f0f025b0a4f3e6ca625`, .NET 10 | 9 | 9 | 0 |
| Current generator restored, .NET 10 | 18 | 0 | 0 |
| Current generator, .NET 8 | 18 | 0 | 0 |
| Current generator, .NET Framework 4.7.1 | 18 | 0 | 0 |

All nine before failures identify the absent positive factory: three syntax
assertions and six generated-source compilations reporting only the missing
abstract factory implementation. These are regression-guard failures for the
new generator/base contract, not nine claimed production detector defects.
The nine assertion controls still pass against that prior generator.

Final result files in `artifacts/pr2154-review/`:

- `pr2154-positive-text-complete-before.trx`
- `pr2154-positive-text-complete-restored-net10.trx`
- `pr2154-positive-text-complete-net8.0.trx`
- `pr2154-positive-text-net471-final.trx`
- `pr2154-positive-root-independent18.trx` — independent reviewer replay:
  18 passed, zero failed/skipped, .NET 10, 13 seconds.

All three focused project builds completed with zero warnings and zero errors.
An earlier EAST oracle mistakenly expected a four-dimensional public result;
the actual documented flattened contract was checked and corrected before the
final runs. Earlier harness failures concerning synchronous timeout support or
missing trace-helper references are not included as before/after proof.

The generator's normal metadata discovery runs against real production model
types. This small discovery compilation omits the repository's unrelated manual
test census, so its global coverage/name-collision diagnostics are printed, not
claimed as whole-project validation. The test requires exactly the three real
hint identities and compiles **every selected generated source** against the
actual shared test base and actual detector assemblies with zero emit errors.
It then invokes the generated classes. No mock detector/base or handwritten
replacement leaf supplies the proof.

SHA-256 identities:

| Assembly | SHA-256 |
| --- | --- |
| Prior generator | `488763248BE3B050EB1EACB8BAE44CBAECAE80FCC67797C6A436B83D22993A53` |
| Current generator | `7F99F0DA179EB2C184AEAFCA88BEBD9146DF8C0C16437D72E1DFC76A6D3A80F2` |
| Actual unchanged .NET 10 core | `69BCCF0B1B5AD45EB4050C3D83D352980BB7F0F8EC4D3153EBEBB5469A4B9B8D` |
| Actual unchanged .NET 8 core | `97F5D10424D8B75698A144E248797B18A149F08B376727361A37E609E82B27BA` |
| Actual unchanged .NET Framework core | `B6E19F6F44C9F6E43FFA2A9A29B14B9999B9F76F86669F273644027CA6BEB64B` |

These are the previously validated shared-layout core binaries; this batch
changes only the scaffold generator and tests. The frozen AP and shared-layout
review runner outputs were not rebuilt or overwritten.

## Reproduce from the repository root

First build the current generator and actual Release core for each framework,
or reuse the hash-identified core outputs above. The preceding
`Pr2154.DetectionParameters` proof documents the existing validated CPU native
closure. This runner reuses that closure rather than copying every native RID.
It includes the repository's real module initializer, licensing support,
generated-test trace helper, and xUnit configuration. The imported managed-only
closure target supplies .NET Framework dependencies without a native-tree copy.

```powershell
$project = 'review-tests/Pr2154.PositiveDetections/Pr2154.PositiveDetections.csproj'
$env:PATH = (Resolve-Path -LiteralPath 'review-tests/Pr2154.DetectionParameters/bin/Release/net10.0').Path + ';' + $env:PATH
dotnet restore $project
dotnet build src/AiDotNet.Generators/AiDotNet.Generators.csproj -c Release --no-restore -p:GeneratePackageOnBuild=false
foreach ($framework in @('net10.0', 'net8.0', 'net471')) {
    dotnet build $project -f $framework -c Release --no-restore -m:2 -nodeReuse:false -p:BuildProjectReferences=false -p:GeneratePackageOnBuild=false
    if ($LASTEXITCODE -ne 0) { throw "Build failed for $framework" }
    dotnet test $project -f $framework -c Release --no-build --no-restore --logger "trx;LogFileName=positive-text-$framework.trx" --results-directory artifacts/pr2154-review
    if ($LASTEXITCODE -ne 0) { throw "Tests failed for $framework" }
}
```

To reproduce the isolated prior-generator control, supply its exact DLL through
`-p:GeneratorAssemblyPath=<absolute DLL path>` on the focused build. Verify its
SHA-256 against the table before running the same 18 tests; expect nine failures
and nine passes, not a green baseline. Omitting that property on the subsequent
build restores the current generator; verify its copied hash and rerun green.
The property changes only this runner's reference and does not modify either
generator binary or the actual core.
