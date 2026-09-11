# PR #2154 bounded review validation

This project compiles the real AiDotNet library and generator and source-links the changed regression tests, the relevant existing numerical/metrics tests, and all four edited detection/OCR model-family bases. It does not stub production contracts or replace generated family fixtures.

The baseline is PR head `ebf7a1c9891af791e8715fb4bc5c74c1b270c34a` (base `1c8647e293ff9f5180a071a8e42f16dc90849102`). The reviewed changes are a local follow-up, not a claim that the whole PR is merge-ready. The corrected exhaustive inventory contained **36 threads, 34 unresolved**; thread pagination and every per-thread comments connection reported `hasNextPage: false`. The earlier 33/31 inventory was incomplete, not evidence that three threads had been resolved.

## Reproduction

Run from the follow-up worktree using PowerShell and the installed .NET 10 SDK. Enter the path to your own clean baseline worktree when prompted; the guard rejects a missing path, a different commit, or uncommitted files before starting a build. These commands reproduce the recorded 175-case inventory, excluding the later text-input cases documented separately below. This focused project uses the repository's real `ModuleInitializer.cs`, licensing test support, `GlobalUsings.cs`, and `xunit.runner.json`; no numerical tolerance is relaxed. CPU selection is test-only. Runtime preprocessing still uses the selected tensor engine, with no CPU-only production switch.

```powershell
$baselineRoot = Read-Host 'Path to the clean PR #2154 baseline worktree'
if ([string]::IsNullOrWhiteSpace($baselineRoot)) {
    throw 'A baseline worktree path is required.'
}
$baselineRoot = (Resolve-Path -LiteralPath $baselineRoot -ErrorAction Stop).ProviderPath
$expectedBaselineHead = 'ebf7a1c9891af791e8715fb4bc5c74c1b270c34a'
$baselineHead = git -C $baselineRoot rev-parse --verify HEAD
if ($LASTEXITCODE -ne 0 -or $baselineHead -ne $expectedBaselineHead) {
    throw "The baseline must be checked out at exactly $expectedBaselineHead."
}
$baselineChanges = @(git -C $baselineRoot status --porcelain)
if ($LASTEXITCODE -ne 0 -or $baselineChanges.Count -ne 0) {
    throw 'The baseline worktree must have no tracked or untracked changes.'
}
$baselineSourceRoot = Join-Path $baselineRoot 'src'
$baselineProject = Join-Path $baselineSourceRoot 'AiDotNet.csproj'
if (-not (Test-Path -LiteralPath $baselineProject -PathType Leaf)) {
    throw 'The baseline worktree does not contain src/AiDotNet.csproj.'
}
$env:AIDOTNET_FORCE_CPU='1'
dotnet build $baselineProject -c Release -f net10.0 -p:GeneratePackageOnBuild=false
if ($LASTEXITCODE -ne 0) { throw 'The baseline build failed; do not test a stale DLL.' }
dotnet test review-tests/Pr2154.ComputerVision/Pr2154.ComputerVision.csproj -c Release "-p:ReviewedSourceRoot=$baselineSourceRoot" -p:BuildProjectReferences=false -p:GeneratePackageOnBuild=false --filter 'FullyQualifiedName!~Pyramid_RejectsOverflowingStrideWithoutEnteringLegacyShiftLoop&FullyQualifiedName!~CvInputBoundaryReviewTests.TextDetector_' --logger 'trx;LogFileName=pr2154-boundary-full-baseline.trx' --results-directory artifacts/pr2154-review --verbosity quiet '-clp:ErrorsOnly' '-flp:logfile=pr2154-boundary-full-baseline.log;verbosity=normal'

dotnet test review-tests/Pr2154.ComputerVision/Pr2154.ComputerVision.csproj -c Release -p:GeneratePackageOnBuild=false --filter 'FullyQualifiedName!~CvInputBoundaryReviewTests.TextDetector_' --logger 'trx;LogFileName=pr2154-boundary-full-after.trx' --results-directory artifacts/pr2154-review --verbosity quiet '-clp:ErrorsOnly' '-flp:logfile=pr2154-boundary-full-after.log;verbosity=normal'
```

The baseline source lives in a separate, clean, detached worktree at the exact head above. `ReviewedSourceRoot` changes only the real library/generator project references; both runs compile the same final test sources. Do not run the commands concurrently: the focused project's output directory is intentionally shared. Within the recorded 175-case inventory, the baseline excludes only two stride values above `2^30`: the legacy signed left-shift loop does not terminate for them. These tests are not skipped in source or CI and execute in the unfiltered follow-up run.

## Text-input follow-up evidence (current 208-case inventory)

Comments **3991259128** and **3991259119** add one shared text-detector input validator and portable reproduction inputs. The text-detector fix is at the shared base, not in concrete detectors or generated leaf tests. Both consuming paths validate a snapshot of the publicly mutable `InputSize` array before indexing, resizing, or allocating a deferred input. The already-resolved serialization path does not consume the option and still returns without another forward pass.

The 33 added cases cover prediction, preprocessing and initial serialization with null external bindings, empty/short/long arrays, zero/negative dimensions, valid `1x1`/`2x3` inputs, normalization and input nonmutation, repeated serialization, and in-place dimension mutation after a real prediction. The **before** library is the unchanged review head `f74c1a6d5c80b197f22ec2d2c4f76895c5ff5762`; both runs compile the same final test sources. To reproduce this newer baseline, use that exact commit as `$expectedBaselineHead` in the guard above and omit the historical `--filter` arguments. That head already contains the stride-overflow fix, so no case needs exclusion.

| Run | Passed | Failed | Skipped | TRX under `artifacts/pr2154-review` |
| --- | ---: | ---: | ---: | --- |
| Unchanged `f74c1a6d5c` DLL, 33 new cases | 7 | 26 | 0 | `pr2154-text-boundary-before.trx` |
| Unchanged `f74c1a6d5c` DLL, unfiltered suite | 182 | 26 | 0 | `pr2154-text-boundary-full-before.trx` |
| Shared validator, unfiltered suite | 208 | 0 | 0 | `pr2154-text-boundary-full-after.trx` |
| Fresh-process no-build repeat | 208 | 0 | 0 | `pr2154-text-boundary-full-after-repeat.trx` |

All original 175 controls passed before and after. Before the fix, null/short arrays produced `NullReferenceException`/`IndexOutOfRangeException`, negative dimensions reached an `OverflowException`, and long arrays were accepted. These are 26 failing cases for one missing shared validation boundary, not 26 distinct defects. The new guard consistently reports `ArgumentException` with `ParamName == "InputSize"` before forwarding. The selected-engine resize/multiply path and strict pixel/gradient controls are unchanged; these CPU runs are not physical-GPU proof.

The final core build completed with **0 errors, 2,842 warnings**, in 4m29s. The unfiltered before/after runs took five seconds each; the fresh-process repeat took four seconds. The loaded focused-project DLL hashes were captured before subsequent builds could replace them:

| Artifact | Before SHA-256 | After SHA-256 |
| --- | --- | --- |
| `AiDotNet.dll` | `52D186AFC8B5484E5A131D39CC060C2E27833F11A7A0FFB1E51BFD08353F2C7F` | `B254FAD5B62ABC52973F4634AD50B30343EF4A64E34A83F4B454280008556FD3` |
| `AiDotNetTests.dll` | `F45C6701F6837599D9BE1A69346FA4F65E6B049826CAF68B2F04B5AD7D5132A7` | `9E36E347EC2C93DC22640B8B142FEDC31B0DC0B7EEAA2FC36918A551A8D0DD02` |

`AiDotNet.Tensors.dll` remained `EB681AE60F23B03CF08E0BF3AB70A372673927ACD87A428C74536D424846D5E7`. The documented PowerShell block parsed without errors; its guard accepted the actual clean `ebf7a1c989` baseline and rejected empty input, a missing path, and the wrong-head review worktree. No build was launched by those guard-only checks.

The actual unfiltered follow-up commands, after building the current core, are:

```powershell
dotnet test review-tests/Pr2154.ComputerVision/Pr2154.ComputerVision.csproj -c Release -p:BuildProjectReferences=false -p:GeneratePackageOnBuild=false --logger 'trx;LogFileName=pr2154-text-boundary-full-after.trx' --results-directory artifacts/pr2154-review --verbosity quiet '-clp:ErrorsOnly'
dotnet test review-tests/Pr2154.ComputerVision/Pr2154.ComputerVision.csproj -c Release --no-build --no-restore --logger 'trx;LogFileName=pr2154-text-boundary-full-after-repeat.trx' --results-directory artifacts/pr2154-review --verbosity quiet
```

## Extended boundary evidence (recorded 175-case inventory)

This recorded suite includes the primary reviewer's independent non-integer `3x5 -> 2x2` pixel/gradient oracle and 42 separate input/stride boundary cases. The exact commands are above.

| Run | Passed | Failed | Skipped | Not selected | TRX under `artifacts/pr2154-review` |
| --- | ---: | ---: | ---: | ---: | --- |
| Exact baseline DLL, current test sources | 123 | 50 | 0 | 2 | `pr2154-boundary-full-baseline.trx` |
| Follow-up, unfiltered | 175 | 0 | 0 | 0 | `pr2154-boundary-full-after.trx` |
| Follow-up, fresh-process no-build repeat | 175 | 0 | 0 | 0 | `pr2154-boundary-full-after-repeat.trx` |
| Primary reviewer's independent final replay | 175 | 0 | 0 | 0 | `pr2154-expanded-root-independent.trx` |

The baseline's 50 failures comprise the earlier 17, the independent resize oracle, and 32 boundary-contract cases. Some invalid-stride cases previously rejected the value only inside `Log2` with a singular `stride` parameter error; the new boundary consistently rejects the caller's `strides` configuration before assignment. These counts are cases, not distinct defects. Eight new valid/fast-path controls already passed on the baseline; all 123 baseline-passing controls remain green. The two unselected baseline cases are `int.MaxValue` and `2^30 + 1`, whose legacy loop cannot terminate; their unfiltered after results are passing, not claimed before executions.

That source build reported **0 errors, 2,775 warnings**, 3m43s. Test durations were eight seconds before, five seconds after, and four seconds for the repeat. The test sources and numerical tolerances were identical across the baseline and follow-up compilations. The new guard preserves the already-resolved serialization return, and valid stride boundaries include both `1` and the largest positive signed-int power of two (`2^30`).

| Artifact | Baseline SHA-256 | Follow-up SHA-256 |
| --- | --- | --- |
| Loaded `AiDotNet.dll` | `9114068D81C131953BBA5087943181725C0046348E61DEFA3CDAA433E136C2DD` | `52D186AFC8B5484E5A131D39CC060C2E27833F11A7A0FFB1E51BFD08353F2C7F` |
| `AiDotNetTests.dll` | `5FB685DE4C67610C78BE4374FFE27BC25059CF11E9F81EAF9CA02ED8C8298221` | `DFC40A44A69B4B74705C7A1A984364393FFBEDE9BF8E8370E8188689A09A09C3` |
| Boundary TRX | `A6479CF7DCF80A27668818FBC4D69C62787D039914AD8BF77B1BC6A6A05BE48E` | `B98C784F9149C8D62FA4427B39510FB3358AEFB01E9BC34A15DB12AE9D1E5B3F` |

The loaded tensor dependency is unchanged (`AiDotNet.Tensors` 0.130.3; SHA-256 `EB681AE60F23B03CF08E0BF3AB70A372673927ACD87A428C74536D424846D5E7`). The repeat used `dotnet test` with `--no-build --no-restore` and the repeat TRX name, so it did not rebuild or change the tested library.

## Initial corrected-harness evidence (132 cases, historical)

Both initial corrected-harness runs used the same test sources and the published `AiDotNet.Tensors` package `0.130.3`. SDK: `10.0.401`, Windows x64, `net10.0`.

| Actual source | Passed | Failed | Skipped | Source build | TRX under `artifacts/pr2154-review` |
| --- | ---: | ---: | ---: | --- | --- |
| Exact PR baseline | 115 | 17 | 0 | 0 errors, 2,842 warnings; 3m48s | `pr2154-review-final-harness-baseline.trx` |
| Follow-up source | 132 | 0 | 0 | 0 errors, 2,775 warnings; 3m53s | `pr2154-review-final-harness-after.trx` |

Each test run reported a five-second test duration, separate from source compilation. These are 17 failing regression **cases**, not 17 distinct root causes. All 115 baseline-passing controls remained green after the changes. The 132 cases comprise 19 new CV regressions/controls, four actual nullable-generator/runtime cases, and 109 existing numerical, FPN/PANet, TrOCR-decoding and metric cases. The four edited abstract family bases compile but are not counted as executed generated model families.

Concrete observations pinned by the before/after tests include:

- CTC/attention with a two-character budget: baseline emits `abc`; follow-up emits `ab`. Leading CTC blanks/repeats and attention EOS are also checked.
- Multi-image empty RoIs: baseline throws `ArgumentNullException` while concatenating no tensors; follow-up returns shape `[0,3,2,2]`.
- Concurrent warm-up: baseline produced seven duplicate-key exceptions and one silent empty-registry success; follow-up initializes once and reports the real empty-registry error to all eight callers.
- Text prediction: baseline skips the configured spatial resize and gives input gradient `1`; follow-up uses the expected shape and gradient `1/255`, with unchanged input pixels.
- Tensor-list/adapter chunk IDs: baseline layout `weights` does not match `weights/0.0` and `weights/1.0`; follow-up IDs and live storage identity agree. Flat snapshots are not advertised as writable model storage, and mutating them does not change the model.
- The ready-training control passes both before and after: weight `2 -> 1.6 -> 1.28`, losses `4` then `2.56`, and exactly three forward calls (one warm-up plus two gradient steps).

SHA-256 hashes were read from the **loaded focused-project output**, before the next run replaced it:

| Artifact | Baseline SHA-256 | Follow-up SHA-256 |
| --- | --- | --- |
| `AiDotNet.dll` | `9114068D81C131953BBA5087943181725C0046348E61DEFA3CDAA433E136C2DD` | `0C7E3196109DE670DE47C7F7DC6828D0E6700A3C06F01327176C40375EB8A23F` |
| `AiDotNetTests.dll` | `DA3FA766CDA03FB24B63890FB50BB1254A64F0D74F6C526361C0B12FA63A6FA5` | `CE99EC7801BC612958C846ED1E596511C058191BAD1BEA0EAE464CD6A5E7D39C` |
| Final TRX | `7CAB5D99F44A52D8527E11DDDFD0DD43BC637A9232E7975FB4FF6EB97C100961` | `2772A2227F0C706520558DC0E7BCD549D1C25CCF45095B8810EC9AAD96A6930B` |

The `AiDotNet.Tensors.dll` hash is identical in both runs: `EB681AE60F23B03CF08E0BF3AB70A372673927ACD87A428C74536D424846D5E7`. Earlier diagnostic runs remain in `artifacts/pr2154-review`, but are not the final before/after claim.

A fresh-process no-build repeat also passed **132/132**, with no skipped tests, in four seconds: `pr2154-review-final-harness-after-repeat.trx`. This was the initial 132-case inventory, before the independent resize oracle and boundary cases below. `git diff --check` passed; the added C# lines and new C# files contain no null-forgiving operators.

The primary reviewer independently inspected the production/test diff and repeated the
same final binary in another fresh process: **132 passed, zero failed, zero skipped**,
five seconds (`pr2154-review-root-independent.trx`). The loaded `AiDotNet.dll`
SHA-256 matches the follow-up hash above. This replay does not expand the tested scope
to the complete generated model families or GPU execution.

The corrected initialization intermediate run, `pr2154-review-exact-initializer-before-fallback.trx`, passed 129 tests and failed 2 (0 skipped). Both failures were copied parameter payloads incorrectly marked `IsWritableInPlace`, after the separate chunk-ID fix. Those failures directly justify the two shared fallback metadata corrections.

### Diagnostic mistakes explicitly excluded from defect counts

- The first isolated harness omitted the repository's CPU initialization. That produced float-sized discrepancies in strict double numerical tests; importing the real initializer/configuration made those controls pass without source or tolerance changes. `AIDOTNET_FORCE_CPU` alone was not an equivalent setup.
- The original nullable-generator probe used the global namespace. The generator emitted an invalid namespace for that unrelated configuration. The final probe uses an explicit namespace and executes real generated component adapters in four enabled/disabled and optional/required cases. The existing global-namespace generator limitation is not fixed here.
- An initial text-output oracle expected a flattened tensor even though the existing single-output contract preserves rank. The final oracle uses `[1,3,2,2]`.
- An intermediate CRNN cleanup removed `_sequenceFeatureDim` before its weight-file header consumers were checked. The resulting three `CS0103` errors were introduced during this work, not baseline errors. The field and its value `512` are preserved; its persistence role is documented. Unused duplicate scratch state was removed instead.

## Review mapping

IDs below are GitHub review-comment database IDs; the corresponding full thread IDs were retained in the review inventory. The mapping distinguishes completed fixes from the explicitly pending items below; it is not a claim that every review item is complete.

| Comment IDs | Scoped disposition and evidence |
| --- | --- |
| 3985472465 | Both base decoders honor the emitted-character budget. CTC blanks and repeated timesteps do not consume it; zero-budget, blank/repeat and EOS controls are included. |
| 3985472468, 3990067342 | Empty live trainable discovery throws with model identity. A typed shared warm-up gate prevents duplicate initialization, retries failed initialization, and preserves the ready-model fast path. Tests include concurrent first calls, failure retry and two real gradient steps with exact expected weights. |
| 3990067371 | Tensor-list layouts and live chunks use identical stable IDs. Accessor/collection passthrough requires compatible layout metadata; fallback payloads are explicitly not writable model storage. Tests cover live identity, both adapters, flat-only sources and snapshot nonmutation. |
| 3990067051 | Empty RoIs preserve `[0,C,outH,outW]` for single- and multi-image inputs without concatenating an empty list. |
| 3990067177 | Text prediction uses the same asymmetric resize and normalization as detection. The implementation uses tensor-engine operations and retains the input gradient. Exact pixel, input-nonmutation and gradient tests are included. |
| 3990067093 | All three CV base copy-preparation paths replay batch one without modifying the source shape; text and OCR runtime probes cover the shared behavior. |
| 3990067103 | A shared object-detector base guard validates exactly two positive input dimensions before deferred probing or preprocessing. Tests cover null external binding, empty/short/long arrays, nonpositive dimensions, valid 1x1/2x3 inputs and the already-resolved fast path. |
| 3991259128 | The shared text-detector base now applies the same exact-two-positive-dimensions contract to prediction/preprocessing and deferred serialization. The 33 new cases reproduce 26 failures on `f74c1a6d5c` and pass with the private validator; all 175 earlier controls remain green. |
| 3991259119 | Reproduction prompts for the caller's baseline root and validates its exact recorded commit, clean Git state and real project path before building. Both commands use that resolved input rather than an author-specific path. Guard-only positive and negative controls are recorded above. |
| 3990067121 | Shared FPN assignment validates nonempty, positive power-of-two, contiguous doubling strides before taking logarithms. The integer logarithm shifts its value down, so it cannot wrap a left-shift count indefinitely. Tests cover invalid assignment/pooling inputs and valid/invalid signed-int boundaries. |
| 3990067140 | Do not internalize `RPN<T>`: it was already public at merge base `1c8647e293ff9f5180a071a8e42f16dc90849102`, so that recommendation would break an existing public type. Its explicit parameter members already delegate to the shared internal `DelegatingCvParameterModule`; the forwarding does not duplicate the implementation. |
| 3990067157 | Both YOLO head decoders hoist `scaleX`/`scaleY` once per level, using the existing feature dimensions and preserving the arithmetic. Source inspection and real library compilation verify this cleanup; no dedicated decode-speed benchmark is claimed. |
| 3985472484 | Corpus CER/WER return NaN for nonzero edits over zero reference length, consistent with the existing per-sample contract; zero-edit and ordinary-reference controls remain. |
| 3985472480 | Polygon conversion is internal. Existing text-detection metric tests compile and execute against the actual implementation. |
| 3985472488, 3985472504, 3985472513 | Shared family bases verify clone mutation, batch box coordinates and exact source image dimensions. Their real sources compile in this harness; complete generated model families have not been rerun here. |
| 3990067400, 3990067419 | Input dependence always reaches an assertion, including length differences; NMS invariants honor the fixture's typed/overridable threshold properties. No generated leaf tests were edited. |
| 3990067450, 3990067474 | FPN pooling uses an independent exact level oracle; PANet adds the meaningful level-zero pathway case. Existing numerical/pathway tests execute unchanged except these stronger test inputs/oracles. |
| 3990067066, 3990067079, 3990067167, 3990067215 | Stale XML parameter/summary tags and unreferenced GELU helpers are removed. Actual source compilation and repository-wide caller search validate the cleanup; no public detector API is substituted. |
| 3990067246, 3990067275, 3990067306, 3990067323 | Remove unreachable grayscale branching, unused duplicate CRNN scratch state, unused private shape arguments/locals and dead helper code. CRNN's persisted feature dimension remains. Existing TrOCR incremental/full-decoder parity controls execute. |
| 3990067038 | No readiness weakening: actual Roslyn/generator/runtime tests show explicit `?` remains optional under either nullable context, while unannotated components remain required and raise `ParameterLayoutNotReadyException`. |
| 3990067227 | The current concrete TrOCR `Train` override uses teacher-forced logits and cross-entropy, not the base inference `NoGrad` path. This is source-backed rejection of that specific premise, not proof that every real-model training case is correct. |

## Explicitly pending / not claimed

- **3985472460:** proper detector-specific annotation/loss training remains an architectural gap. Concatenating all head outputs prevents dropping heads, but is not proof that generic MSE is a correct detector loss.
- **3985472491 and 3985472507:** deterministic non-empty object/text detection fixtures still need a shared base/generator design. Random-image tests may produce no detections; compiling stronger invariants does not prove their nonvacuity.
- **3985472478:** caching threshold-independent AP data remains a performance follow-up. Per-threshold greedy matching must remain independent; this batch does not replace it with a shared match set.
- This is focused local Windows `net10.0` validation, not the complete CI workflow, other target-framework builds, GPU execution or a full generated model-family sweep. The existing project emits many warnings. GitHub readiness/merge decisions must retain these limitations and the pending architecture work.
- No package-version change or merge-ready claim is part of this batch. The PR remains draft while the explicitly pending work is incomplete.
