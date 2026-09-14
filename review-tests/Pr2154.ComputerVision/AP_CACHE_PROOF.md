# PR #2154: threshold-independent AP work

This follow-up addresses [review comment 3985472478](https://github.com/ooples/AiDotNet/pull/2154#discussion_r3985472478), against exact head `857613ba8c56ab072bf5c72a0300a24caf754feb`. It does not close the separate architecture/positive-fixture findings or claim that the entire draft PR is merge-ready.

## Implementation and adversarial constraints

`ObjectDetectionMetrics` prepares each class's per-image ground truth and stable confidence ranking once per call. The existing single-threshold AP/full precision-recall paths share that preparation; the original prediction list and sort-index array are retained without an additional sorted tuple-array copy.

Range evaluation processes at most **32 thresholds per batch**, with independent greedy claim sets. The prediction rank is the outer matching loop. A lazily computed IoU row is shared inside the batch, and rank stamps distinguish uncomputed entries from zero/NaN values. A claimed candidate is skipped before reading geometry: a malformed prediction that no remaining threshold needs is still not inspected. Strict `>` best-IoU selection preserves first-candidate ties and prevents zero overlap from matching even at threshold zero.

Only true-positive precision/recall points are retained for AP. False positives cannot improve precision at their unchanged recall; the preceding true-positive point dominates them. Initial false positives contribute zero. Consequently the same 101-point interpolation is preserved, while the **public raw precision-recall curve still contains every prediction**. Per-class and per-threshold addition orders are unchanged, including the handling of classes without valid ground-truth boxes.

For one class with `P` predictions, `G` ground-truth boxes, and `Gmax` boxes in its largest image, matching scratch space is bounded by `O(B*G + B*min(P,G) + Gmax)`, where `B <= 32`. Prepared inputs are retained once per class. There is no `P*G` IoU matrix and no collection of matching states proportional to the entire threshold range. COCO's ten thresholds fit in one batch; larger ranges may recompute an IoU once **per batch**, not necessarily once across the whole call.

Non-finite ranges/steps and threshold counts that exceed the original Int32 count representation now fail explicitly. There is no arbitrary threshold-count cap. Boundary tests include 31, 32, 33 and 65 thresholds.

## Failure-first correctness evidence

The 63 new cases exercise preparation counts, independent claims at different thresholds, stable confidence/IoU ties, lazy malformed-box handling, zero overlap, null image/detection filtering, undefined classes, no true positives, finite/count boundaries, and **27 seeded exact ordered-mean comparisons** across nine ranges. Cases use actual `Detection<double>` and `BoundingBox<double>` instances, not substituted matching or numeric providers.

| Run | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Unchanged baseline, final 63-case AP fixture | 51 | 12 | 0 |
| Current production, full focused inventory, net10.0 | 271 | 0 | 0 |
| Current production, full focused inventory, net8.0 | 271 | 0 | 0 |
| Current production, full focused inventory, net471 | 271 | 0 | 0 |

Six before failures expose repeated preparation at 2/10/31/32/33/65 thresholds. The other six expose NaN bounds/step, positive-infinite step and overflowing/infinite counts. All matching/score controls passed before the cache change. Logs/TRXs are under `artifacts/pr2154-review`:

- `pr2154-apcache-expanded-before.trx`
- `pr2154-apcache-final-focused-net10.trx`
- `pr2154-apcache-final-focused-net8.trx`
- `pr2154-apcache-focused-net471.trx`

All three production target frameworks and the full main test project compiled with zero errors. The main test-project compile disables `CopyLocalLockFileAssemblies` to avoid duplicating unused native runtime trees; runtime execution uses the focused runner's actual dependency closure. The runner includes the repository's real CPU/module initializer, licensing support, global usings and xUnit configuration. Its language version now matches the main test project. Compatibility compilation exposed one existing text-input assertion that relied on an xUnit span overload unavailable on net471; changing `result.Shape` to `result.Shape.ToArray()` preserves the exact integer-dimension assertion. The net10.0 full-project compile preceded that test-only representation adjustment; the net8.0/net471 full-project compiles and final focused runs on all three frameworks include it. Production source did not change between those checks. Production whitespace verification and `git diff --check` passed.

The unchanged 63-case AP fixture has SHA-256 `CBD5803B4126609352842438AF5C172239F7BE3CDFD3A92477BF2148803A0712`. Final focused test DLL hashes are `0C6FA8EB44362F98FF615DA565879545937654DC9B2B2AA8151D06BFCB1A24D6` (net10.0), `D7D71A314AEFB496A4126D63495A3B52B0C354A5AEDB9EAB655A1321D84C62AA` (net8.0), and `0F9E7B57FFC473DEE393E427FB9C9893D5AB7F43894252D1660CCB9B1419BB33` (net471).

## Unmodified-production CPU measurements

The same compiled benchmark executable, process/runtime configuration, immutable seeded input and dependency files were used in both arms. Only `AiDotNet.dll` was exchanged between completed processes. Each workload had a one-second warmup and nine measured calls. The two pairs ran in reverse order: after/before, then before/after. Every score/checksum was checked for exact double-bit equality between arms; both pairs matched all seven workloads.

Input: seed 2154, 12 images, four classes, 24 ground-truth boxes per image/class, 3,456 predictions and 1,152 ground-truth boxes. The ten-threshold case is actual COCO `.50:.95` with step `.05`; the 32/33/65 cases use step `1/128`. CPU selection occurs in a module initializer. The engine may probe GPUs during its own static initialization before resetting to CPU; that startup is outside all warmup/measurement intervals. These are **CPU metric microbenchmarks, not GPU or detector-pipeline speedups**. Other system activity was not globally controlled, and there are no flaky wall-clock assertions.

| Workload | Pair 1 median ms, before → after | Pair 2 median ms, before → after | Allocated bytes/call, before → after |
| --- | ---: | ---: | ---: |
| Single mAP | 2.1973 → 1.7263 | 1.6921 → 1.7195 | 389,392 → 388,328 |
| Full raw PR curve | 0.4759 → 0.3900 | 0.5453 → 0.4270 | 90,208 → 89,952 |
| One-threshold range | 2.3092 → 1.8238 | 2.6075 → 1.7440 | 389,352 → 333,744 |
| COCO, ten thresholds | 32.2351 → 5.5906 | 33.2607 → 8.1358 | 3,893,520 → 540,688 |
| 32 thresholds | 95.2698 → 14.9482 | 105.8495 → 15.3530 | 12,459,264 → 1,114,216 |
| 33 thresholds | 103.7252 → 18.1369 | 99.6430 → 17.9884 | 12,848,616 → 1,138,760 |
| 65 thresholds | 167.3811 → 32.7786 | 223.7872 → 34.6994 | 25,307,880 → 1,810,304 |

Single-mAP timing varied slightly in the second pair; this is not evidence of a universal speedup for that unchanged matching path. Its geometric workload is identical and allocation decreases. COCO allocation decreases about 86%; both measured pairs show a substantial range-evaluation speedup.

Raw logs are `pr2154-apcache-pair{1,2}-{before,after}.log`. Important SHA-256 identities:

| Artifact | SHA-256 |
| --- | --- |
| Before production `AiDotNet.dll` | `B254FAD5B62ABC52973F4634AD50B30343EF4A64E34A83F4B454280008556FD3` |
| After production `AiDotNet.dll` | `2D5355184C81886DA018076141A8A1030F9E5C49527C8DDB145B75253399388C` |
| Identical benchmark DLL in all four arms | `64F43C5DAAF3BD7B97FBB97E05D6F6E27A43906A94D5F6F88AC41B09124AC3C3` |
| Benchmark `Program.cs` | `568FBA45B617F08C73FBE978C30F8B091D6452360D7FB565480046105E21F963` |
| Unchanged `AiDotNet.Tensors.dll` (0.130.3) | `EB681AE60F23B03CF08E0BF3AB70A372673927ACD87A428C74536D424846D5E7` |

## Separate, source-isolated workload proof

`Pr2154.APRangeWorkload` compiles a separate copy of each actual metrics source with only `box.IoU(candidates[c])` replaced by `WorkloadProbe.CountIoU(box, candidates[c])`. The wrapper increments a counter and calls the real, unchanged `BoundingBox.IoU`; no geometry stub or global numeric-provider mutation is involved. Reversing that replacement was checked against both original sources, normalizing only line endings/trailing whitespace. The timed production DLLs above contain **none** of this instrumentation.

| Workload | Before IoU calls | After IoU calls |
| --- | ---: | ---: |
| Single mAP | 42,786 | 42,786 |
| Full raw PR curve | 10,624 | 10,624 |
| One-threshold range | 42,786 | 42,786 |
| COCO, ten thresholds | 664,717 | 82,575 |
| 32 thresholds | 1,844,209 | 70,393 |
| 33 thresholds | 1,915,261 | 141,445 |
| 65 thresholds | 4,450,747 | 236,281 |

For this input, lazy row reuse gives an upper bound of `3456 * 24 * ceil(thresholdCount / 32)` calls. Applying that guard to the original source rejects all four multi-threshold cases; the current source passes every case. All seven instrumented scores/checksums also match exactly. This workload guard supplies a deterministic negative control independent of timing.

Artifacts: `apcache-instrumented-{before,after}.cs` and `pr2154-apcache-final-workload-{before,after}.log`. Instrumented executable hashes are `778F770A7684C382F4F12D64656ED03CC01372621CE32B03BB34315A2D4C3E53` before and `14C25E01C5C2C027643204E3A39D9C8EE49753FAA033261D7DF71F80ABDC6D4C` after. The counter runner accepts an explicit `MetricSource` path; its local metrics type intentionally overrides the imported one only in that executable.

## Reproduction

Run from this worktree with the installed .NET SDK, sequentially where output paths are shared. The commands below do not rerun the full model-family matrix:

```powershell
dotnet restore review-tests/Pr2154.ComputerVision/Pr2154.ComputerVision.csproj
foreach ($tfm in 'net10.0','net8.0','net471') {
    dotnet build src/AiDotNet.csproj -c Release -f $tfm --no-restore -p:GeneratePackageOnBuild=false
    if ($LASTEXITCODE -ne 0) { throw "Core build failed: $tfm" }
    dotnet test review-tests/Pr2154.ComputerVision/Pr2154.ComputerVision.csproj -c Release -f $tfm --no-restore -p:BuildProjectReferences=false -p:GeneratePackageOnBuild=false --logger "trx;LogFileName=apcache-$tfm.trx" --results-directory artifacts/pr2154-review
    if ($LASTEXITCODE -ne 0) { throw "Focused tests failed: $tfm" }
}
dotnet restore tests/AiDotNet.Tests/AiDotNetTests.csproj
foreach ($tfm in 'net10.0','net8.0','net471') {
    dotnet build tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -f $tfm --no-restore -p:CopyLocalLockFileAssemblies=false -p:GeneratePackageOnBuild=false
    if ($LASTEXITCODE -ne 0) { throw "Full test-project compile failed: $tfm" }
}
```

For before/after reproduction, obtain a clean baseline worktree at exactly `857613ba8c56ab072bf5c72a0300a24caf754feb`. Use the path-input/clean-head guard in the main README, substituting this exact baseline SHA. Build its net10.0 core and pass its `src` directory as `ReviewedSourceRoot` to the focused runner. Run the final AP fixture unchanged with `--filter 'FullyQualifiedName~ObjectDetectionRangeCacheReviewTests'`; the expected original result is 51 pass/12 fail. Then restore the current project references and rerun all 271 cases.

Build `review-tests/Pr2154.APRangeBenchmark/Pr2154.APRangeBenchmark.csproj` in Release after the core build. Retain only the before/current **core DLLs**, not duplicated native dependency trees. Between completed benchmark processes, copy the selected core DLL into the benchmark output directory as `AiDotNet.dll` and run its benchmark DLL; retain the same benchmark/dependency files for every arm. Record hashes and compare all seven workload/threshold identities and score bits, not just timing numbers.

For the separate counter run, make the single replacement described above in isolated source copies, then build `review-tests/Pr2154.APRangeWorkload/Pr2154.APRangeWorkload.csproj -p:MetricSource=<absolute-source-copy-path>`. Copy only the resulting workload DLL beside the benchmark DLL and run it with `dotnet exec --depsfile <benchmark-deps.json> --runtimeconfig <benchmark-runtimeconfig.json> <workload.dll>`. The shared dependency closure resolves real production geometry without another native-runtime copy. Never use these instrumented assemblies for the production timing table.

No fresh remote CodeQL/CI scan or physical-GPU/full-pipeline performance claim is made by this local evidence.
