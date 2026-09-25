# PR #2154: controlled positive object detections

This batch addresses review `3985472491` / `PRRT_kwDOKSXUF86hUmqh`.
It adds a shared positive invariant and a typed factory emitted by the real test
scaffold generator. No generated leaf test or production numerical forward was
edited or substituted. Existing random-input tests continue to permit empty
results and keep their geometry, ordering, and NMS assertions.

## What was proved

The runner discovers the actual nine model types through production metadata,
runs the real generator, selects their exact nine generated source identities,
compiles all nine against the actual shared base classes, and invokes them.
The positive factory uses an explicit Nano/64x64/two-class profile. Ordinary
generated factories retain their previous defaults.

The fixture initializes each real model, requires live writable trainable chunks,
then configures actual weights. `Predict` receives normalized 0/1 pixels;
`Detect` receives the equivalent 0/255 image, matching its public normalization
contract. Each same initialized model is checked at NMS thresholds 1 and 0.45,
then must reject its known scores at confidence 0.99.

| Actual model/profile | Candidates at NMS 1 | Result at requested NMS 0.45 | Independent checks |
| --- | ---: | ---: | --- |
| YOLOv8/v9/v11 | 84 | 1 | Zero 16-bin DFL logits imply distance 7.5; all clipped boxes are `[0,0,64,64]`. Three live classification biases produce 64 scores of 0.5, 16 of 0.75, and 4 of 0.875. |
| YOLOv10 default | 84 | 84 | Its documented NMS-free default is preserved; the candidates remain confidence-ranked. |
| YOLOv10 with explicit `useNmsFree: false` | 84 | 1 | The actual supported NMS mode suppresses duplicates without replacing the numerical forward. |
| DETR / RT-DETR / DINO | 2 | 1 | Two distinct live query directions pass through actual decoder normalization. An independent scalar normalization/softmax oracle includes the background class. Zero box logits imply exact `[16,16,48,48]` boxes. Scores are approximately 0.952574 and 0.658553. |
| Faster R-CNN / Cascade R-CNN | 275 | 39 | A real spatial channel passes through backbone, FPN, ROIAlign, and classifier. Raw heads establish scores/proposals; independent softmax and greedy IoU/NMS check decoded detections and ordering. |

For the R-CNN profiles, zero RPN score/delta heads yield fixed proposals and zero
ROI regression heads preserve proposal geometry. The highest-score proposal is
independently pinned to the stride-4, ratio-1/2 anchor centered at `(62,46)`:
`[62-16*sqrt(2), 46-16/sqrt(2), 64, 46+16/sqrt(2)]`. R-CNN's oracle derives other
scores and proposals from actual raw model outputs; it is **not** a separately
implemented reference backbone or evidence of trained recognition accuracy.

Non-vacuity guards require at least two same-class candidates, genuinely distinct
scores, and an overlap that requires suppression. Independent negative controls
reject empty/missing results, reversed/tied scores, wrong classes/geometry,
non-finite values, wrong image dimensions, ignored NMS, lower-score winners, and
over-suppression. Three additional real-model mutants corrupt only `Detect`
(empty/reverse/ignore NMS), never `Forward`. Each must fail the shared invariant;
`MutationApplied` proves the failure occurred after reaching that corruption,
not at an earlier constructor, live-state, or raw-oracle precondition.

## Failure-before / success-after evidence

The same final 49-case test source was compiled against the previous generator
from `7e1cb86a4ba9a5d4a02569d870d28d26a4a5d0de`, then against the new generator.
Both used the same unchanged actual .NET 10 production DLL. The old generator
misses nine positive factories: nine syntax assertions fail and eighteen runtime
cases fail compilation with the exact nine `CS0534` missing-factory diagnostics.
The other 22 controls pass. Those failures are missing coverage infrastructure,
not fabricated claims that the old production models returned empty results.

| Run | Passed | Failed | Skipped | Report in `artifacts/pr2154-review` |
| --- | ---: | ---: | ---: | --- |
| Final-source historical generator, .NET 10 | 22 | 27 | 0 | `pr2154-positive-object-final-before49.trx` |
| Final .NET 10 | 49 | 0 | 0 | `pr2154-positive-object-final-net10.trx` |
| Final .NET 8 | 49 | 0 | 0 | `pr2154-positive-object-final-net8.trx` |
| Final .NET Framework 4.7.1 | 49 | 0 | 0 | `pr2154-positive-object-final-net471.trx` |
| Independent parent-agent replay, .NET 10 | 49 | 0 | 0 | `pr2154-positive-object-root-independent49.trx` |

The 49 cases are nine factory contracts, nine actual positive invariants, nine
ordinary generated-fixture replays (each invokes three existing invariants),
19 oracle/precondition controls, and three live-model negative controls.
Generator build: zero errors, 67 analyzer warnings. Focused runner builds:
zero errors on all three frameworks (0/2/0 warnings on net10/net8/net471).
No production core rebuild was required for these test/generator-only changes.
The independent replay verified the exact final test-DLL hash and completed in
31 seconds after a separate full source review of the shared fixture and controls.

SHA-256 identities:

| Artifact | SHA-256 |
| --- | --- |
| Historical generator | `7F99F0DA179EB2C184AEAFCA88BEBD9146DF8C0C16437D72E1DFC76A6D3A80F2` |
| Final generator | `1AF4448E70ED82A2248EDC7077225B7925ED9F69E78CE642331D71BC5587141F` |
| Unchanged net10 core | `69BCCF0B1B5AD45EB4050C3D83D352980BB7F0F8EC4D3153EBEBB5469A4B9B8D` |
| Unchanged net8 core | `97F5D10424D8B75698A144E248797B18A149F08B376727361A37E609E82B27BA` |
| Unchanged net471 core | `B6E19F6F44C9F6E43FFA2A9A29B14B9999B9F76F86669F273644027CA6BEB64B` |
| Final net10 test DLL | `6A5E039CB1106415509DDD8F74F38D3EFBDD9602175E07834CF96A4E8EF72AD8` |
| Final net8 test DLL | `B7A103D563DBBF33C38D7D79B8E23C6D639DF128460CFCA2740B13566B4512D5` |
| Final net471 test DLL | `0943A30D97872B3780B261AF566AFF1E8E1FF795A35BC9D11867875A001B424A` |

## Reproduction

Run from the repository root after the actual production core has been built for
the selected framework. This small runner reuses that output and the installed
CPU native closure; it does not copy all RID/native assets. Its settings explicitly
disable `CopyLocalRuntimeTargetAssets`, `CopyLocalLockFileAssemblies`, and child
project Content propagation. The actual module initializer, license helper,
generated trace helper, and xUnit configuration are included. .NET Framework uses
the existing managed-only test dependency closure target.

```powershell
dotnet restore review-tests/Pr2154.PositiveObjects/Pr2154.PositiveObjects.csproj -p:NuGetAudit=false
dotnet build src/AiDotNet.Generators/AiDotNet.Generators.csproj -c Release --no-restore -m:1 -nodeReuse:false
$framework = 'net10.0' # also validated: net8.0 and net471
dotnet build review-tests/Pr2154.PositiveObjects/Pr2154.PositiveObjects.csproj -f $framework -c Release --no-restore -m:1 -nodeReuse:false -p:BuildProjectReferences=false -p:GeneratePackageOnBuild=false
$cpuNativeDirectory = (Resolve-Path "review-tests/Pr2154.DetectionParameters/bin/Release/$framework").Path
$env:PATH = $cpuNativeDirectory + ';' + $env:PATH
dotnet test review-tests/Pr2154.PositiveObjects/Pr2154.PositiveObjects.csproj -f $framework -c Release --no-build --no-restore --logger 'trx;LogFileName=positive-object-replay.trx' --results-directory artifacts/pr2154-review
```

For an isolated historical comparison, pass the path of a generator built from
the exact historical revision above as `GeneratorAssemblyPath`. Verify it before
using it; the retained previous text-fixture output is the local evidence source:

```powershell
$historicalGenerator = (Resolve-Path 'review-tests/Pr2154.PositiveDetections/bin/Release/net10.0/AiDotNet.Generators.dll').Path
if ((Get-FileHash -LiteralPath $historicalGenerator).Hash -ne '7F99F0DA179EB2C184AEAFCA88BEBD9146DF8C0C16437D72E1DFC76A6D3A80F2') { throw 'Historical generator hash mismatch.' }
dotnet build review-tests/Pr2154.PositiveObjects/Pr2154.PositiveObjects.csproj -f net10.0 -c Release --no-restore -m:1 -nodeReuse:false -p:BuildProjectReferences=false -p:GeneratorAssemblyPath=$historicalGenerator
# Run the same 49 tests; expected outcome: 22 passed, 27 failed, no skips.
# Rebuild without GeneratorAssemblyPath to restore the current generator afterward.
```

This is focused local CPU proof, not a full detector-family/all-repository CI run,
GPU validation, a trained accuracy benchmark, or semantic detector-training proof.
The production training review and newly reported metric/text-boundary findings
remain separate open work; this batch alone does not make the PR ready to merge.
