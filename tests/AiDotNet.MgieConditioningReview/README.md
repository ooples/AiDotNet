# PR2136 MGIE conditioning review

This focused runner references the actual AiDotNet library and shared test initializer.
Its runtime fixtures contain real, small LLaVA, encoder/decoder mapper, U-Net and VAE
components. Recording observers call the real U-Net; they do not supply fabricated
predictions. Generator discovery controls use synthetic declarations only to exercise
structural classification; the runtime factory is separately discovered from the actual
referenced library, semantically compiled against that library, and executed.

## Preserved before evidence

The baseline is the completed Matcha head `5fa1995e69b32cd799a0a67cd05d8202b8a4a627`.
Its actual net10 core SHA-256 is
`A85356AD2042F207CB61D01F749BB7A7C1FFF118B8C4FED54C8BBFB020E5A949`.

- `artifacts/pr2136-mgie-review/pr2136-mgie-before-4.trx`: 2 passed, 2 failed.
  The actual VAE mode/scaling distinction and actual 8-input/4-output-channel U-Net
  controls pass. Public editing fails by feeding four-channel VAE latents into the
  768-wide old edit head; the private denoising probe demonstrates ignored context and
  absent source conditioning.
- Before-four test SHA-256:
  `8C142E2486345E2AE3074760B0DB98547466DAEAA930032A8A299DAA16B1D64A`.
- `artifacts/pr2136-mgie-review/pr2136-mgie-before-9.trx`: 3 passed, 6 failed.
  Additional actual controls expose the batched CLS token rank mismatch, detached
  visual/text concatenation gradients, and reflection traversal selecting frozen child
  weights instead of the owner's declared live parameter storage in either visitation order.
- Before-nine test SHA-256:
  `B97C72BF5C24FB4E7448C01AEA1B2D8E96EB7D318284EF539A98362129785AD9`.

The preserved test assemblies are `mgie-before4-AiDotNetTests.dll` and
`mgie-before9-AiDotNetTests.dll` in that artifact directory. Initial test-compilation
mistakes are not production-defect evidence. The private two-argument denoising probe
was subsequently migrated to the implemented five-argument, three-branch contract;
its final source is not claimed to be the unchanged original probe.

The intermediate foundation build, before integrating the new MGIE path, produced
14 passed and 2 still-failing old-MGIE cases in `pr2136-mgie-foundation-16.trx`.
Core SHA-256: `5E436394E9AC56E00EA5FC1F494B7D80AEBBC4B88D4C023A0CB6C964010A8509`.
Test SHA-256: `1430BEE61CA1B5D89AB922E9FEB353B67F1EC0FD5A1EC01F4492192C887E01A9`.
This is foundation-only evidence, not a successful integrated edit.

## Scope and limits

The native route makes joint image/instruction states and learned edit/query embeddings
trainable, then conditions the real diffusion predictor on both those states and the
unscaled source VAE mode. Three-branch guidance is evaluated in one predictor batch.
An optional typed scheduler capability supplies initial noise sigma and input scaling;
only target noise is scaled, never the appended source channels. Existing unconditioned
generation remains on its original route.

The mapper geometry follows the released [MGIE implementation](https://github.com/apple-aiml-research/ml-mgie/blob/main/mgie_llava.py):
an encoder/decoder transformer with learned queries, not a stack of self-attention-only
layers over VAE latents. Construction here creates native trainable weights. It does
not import released Vicuna/LLaVA checkpoints or identify checkpoint-specific `[IMG]`
token IDs, and random-weight outputs are not useful-edit or image-quality proof.
Existing LLaVA text generation is reused by the optional expressive string route.

Vision encoder geometry and diffusion output geometry are separate options. The default
joint sequence capacity is raised from the inherited 512 to 2048 because the actual
512-pixel/14-pixel-patch visual sequence, CLS, text and edit tokens do not fit in 512.
The bounded generator fixture does not reduce production widths, depths or image defaults.

All results in this document are local CPU correctness evidence unless explicitly stated
otherwise. Engine-based operations preserve the regular device-capable path; that alone
is not an actual GPU execution, throughput or pretrained-quality benchmark.

The remaining [MGIE review thread](https://github.com/ooples/AiDotNet/pull/2136#discussion_r3975251968)
maps to these actual-model controls, rather than helper-only demonstrations:

| Concern | Direct regression evidence |
| --- | --- |
| Instruction and source image must both affect joint states | `MgieJointMapperTests.JointContext_DependsOnBothModalities_AndIncludesRegisteredRawState` and `MgieSamplingPhysicsTests.FixedNoise_ActualDenoisingRespondsToBothSourceAndContext` |
| Source-conditioned diffusion must use the right coordinates at every step | `MgieSamplingPhysicsTests.ActualUnet_AllStepsMatchThreeBranchGuidanceAndEulerUpdate` and the source/noise nonmutation controls for both scheduler capability routes |
| Learned edit/query tokens and nested language weights must participate in training and persistence | `MgiePersistenceAndTrainingTests` checks real noise-loss gradients, roundtrip state restoration and independent clone storage; `JointVisionLanguageStateTests` exercises authoritative nested ownership |
| Image/latent/context geometry and caller lifecycle must be valid | `MgieJointMapperTests.RealEditing_PreservesImageRank_AndModes_OnSuccessAndFailure` and `VisionResolutionIsIndependentOfDiffusionResolution`, plus the semantically compiled actual generator factory |

## Attention dependency and integrated intermediate result

The actual MGIE mapper exposed a shared attention validation defect: noncausal
cross-attention rejected three queries against two keys. The corresponding Tensors
fix is [draft companion PR1035](https://github.com/ooples/AiDotNet.Tensors/pull/1035),
commit `04a4f76607d5e98f0a95fc3263a120453a0dafc7`, based on fresh main
`5d22aa7f4aec3f0a2d0b14953d3d398333f376dd`. It changes only three matching validators;
causal and explicit nonzero-offset fitting bounds remain intact and overflow-safe.
The framework bridge uses the same predicate. No fused path is disabled.

The companion's final 34 new tests produced 17 real failures against its original
core; 34 new plus 58 existing tests then passed on net10, net8 and net471, with an
independent 92-case net10 replay. These are CPU value/gradient tests, not physical-GPU
performance proof. The original framework/published-package control here was
`pr2136-attention-rectangular-before.trx`: three passed, five failed. Its failures
were four rectangular-query rejections and an overflowing positive-offset acceptance.

`pr2136-mgie-local-tensors-net10.trx` is the first integrated replay after both
attention corrections: **44 passed, one failed, zero skipped**. All ten direct
framework/dependency attention cases, the actual generated factory, conditioning
physics, joint gradients, source/context sensitivity and serialization controls passed.
The remaining real failure was cloning: bare `_options` names lost the distinction
between MGIE's options and its diffusion base's options, rejected the carried
constructor arguments, and reconstructed default-sized injected components. The
existing manifest guard correctly rejected that structurally different clone.

For that intermediate run:

- AiDotNet SHA-256: `9A33FF8B6AD3D9DA392C98EA9FAAEE94D0BAA835494946F8C346A1E2E85D05E9`
- Test SHA-256: `05C5DE6E5BA1C74A2E26AFDBF6C877ED7C58EA4E16BC4101D5D0206A752F9DE1`
- Actual loaded local Tensors SHA-256: `CF366B1B208EA6342AF377406CAF67DA4768F54EDD17C6C3BA18B908C59D771F`

The attention tests record the runtime-loaded assembly path and hash in TRX. The local
Tensors DLL was copied only into this owned test runner; the published 0.130.3 package
and global NuGet cache were not modified. A local dependency replay does not satisfy
the eventual stable-package release requirement.

Four bounded real generated-clone controls reproduce the same owner-identity defect
for fields/properties and generic base types supplied as source or referenced metadata:
`pr2136-clone-owner-corrected-before.trx` has **four failures**, all changing a configured
23 to its default 1. The earlier metadata fixture used an incorrect assembly-loading
context; those two loader errors in `pr2136-clone-owner-before.trx` are harness errors,
not additional production defects. An attempted eight-case original-generator replay
did not start because its original generator DLL path was unavailable; it is not
reported as test evidence.

Likewise, the first integrated 35-case run's generator-stub/inventory compilation errors
and a token-permutation oracle were corrected separately. They are not counted as
production defects: the final content-sensitivity test changes actual token content
at its original tolerance, and actual metadata discovery semantically compiles and
executes the emitted real factory.

## Shared clone ownership and adversarial controls

The shared generator now retains the selected declaring member when a constructor
path is genuinely shadowed. The runtime carries that typed identity through generic
owner rebinding and configuration restoration. Non-shadowed plans retain their existing
route, and the public four-argument `ClonePlan` constructor remains available. Explicit
malformed owners fail closed rather than constructing default-sized replacements.
Unavailable null nested paths can select a different recorded constructor. The new
public typed plan snapshots its caller-provided lists, and repeated source options are
restored into each independently cloned destination. The original MGIE clone manifest
check is unchanged.

`pr2136-mgie-owner-integrated-net10.trx` passed all **126 tests**, with zero skips,
including the actual MGIE clone, eight generated owner controls, nine public runtime
binding controls and unchanged parameter-generator/analyzer regressions. This is an
intermediate result, not the final source validation:

- AiDotNet SHA-256: `D5735687EEF4199666771DB01904688D03B73C61670A38928A9FACAEA8ADC205`
- Test SHA-256: `ACC4644F8484239319217AAAE186E467506D9FD545DB5B98029AA14F018C3F41`

An additional adversarial review then found that a partially specified generic owner
could match incompatible fixed type arguments. The same eight final-source controls
against that intermediate core produced **four failures and four positive-control
passes** in `pr2136-clone-partial-before.trx`; each failure was an incompatible owner
incorrectly accepted without an exception. They cover fixed and nested generic
arguments, repeated type parameters, and array shape. The corrected matcher preserves
those constraints while rebinding compatible open type parameters.

- Before test SHA-256: `DEE6F2B5876E2E00580563C7C015F07E3E741E2E8D9DED47FDCEDE0C9ECD02B5`
- Preserved inputs: `clone-partial-before-AiDotNet.dll` and
  `clone-partial-before-AiDotNetTests.dll` in the artifact directory.

## Final integrated validation status

The final recursive generic-owner correction passed the expanded **134-test cohort**
on net10: zero failures and zero skips, 35 seconds,
`pr2136-mgie-final134-premerge-net10.trx`. The actual core rebuilt successfully with
zero errors (2858 warnings, including the rebuilt generator project's existing diagnostics).
The focused runner built with zero errors and four xUnit assertion-style warnings.

- AiDotNet SHA-256: `25BD07BBE516B1CC94CBDB17DA55B2F4DFA4BB613C6400E3BD8E5EBCB7AE7693`
- Test SHA-256: `DEE6F2B5876E2E00580563C7C015F07E3E741E2E8D9DED47FDCEDE0C9ECD02B5`
- Runtime-loaded local Tensors SHA-256: `CF366B1B208EA6342AF377406CAF67DA4768F54EDD17C6C3BA18B908C59D771F`

The test DLL is byte-identical to the eight-case partial-generic failure-before DLL;
all eight now pass within this full cohort. The earlier 126-test result is not being
substituted for validation of this corrected source. This 134-case result is explicitly
**pre-merge**: collaborator commits through `3e517f95670430ca71e11d892045f9d1f46e05a8`
arrived during validation and must be preserved and included in the final post-merge
net10/net8/net471 replay before pushing this batch. The attention package remains an
unpublished companion dependency.

The first final build attempt was interrupted by a full disk while
writing compiler output; its shell exit code does not establish a successful build,
and the core remained the intermediate `D5735687...` assembly. The failed documentation
write also truncated this untracked README, which was restored from the text and
evidence captured immediately before that write. Production/test source was unchanged.
Only three owned inactive output/artifact folders were subsequently NTFS-compressed;
all 438 DLL/TRX hashes were unchanged and nothing was deleted. The successful recovered
core build used `--no-incremental` and fail-closed PowerShell error handling. Post-merge
compatibility results are pending, not inferred from the pre-merge binaries.

## Reproduce the actual-library replay

Run from this AiDotNet checkout after restoring its projects and building the three
matching target frameworks of companion commit `04a4f76607d5e98f0a95fc3263a120453a0dafc7`.
The paths below are the actual local proof locations; adjust the companion checkout
and an existing validated native-library directory for another machine. The native
search path reuses the completed Matcha runner's host libraries without copying its
native dependency tree. These commands intentionally do not modify the global NuGet
package cache or claim that the companion is published.

```powershell
$ErrorActionPreference = 'Stop'
$tensorReviewRoot = 'C:/Users/cheat/source/repos/AiDotNet.Tensors-wt/mgie-rectangular-attention-20260911'
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
$runnerProject = 'tests/AiDotNet.MgieConditioningReview/AiDotNet.MgieConditioningReview.csproj'
$resultsDirectory = 'artifacts/pr2136-mgie-review'
$runId = Get-Date -Format 'yyyyMMdd-HHmmss'
New-Item -ItemType Directory -Force -Path $resultsDirectory | Out-Null

foreach ($framework in @('net10.0', 'net8.0', 'net471')) {
    dotnet build src/AiDotNet.csproj -c Release -f $framework --no-restore `
        -p:GeneratePackageOnBuild=false -p:CopyLocalRuntimeTargetAssets=false `
        -p:CopyLocalLockFileAssemblies=false -m:1 -v:q
    if ($LASTEXITCODE -ne 0) { throw "Core build failed for $framework; do not test a stale DLL." }

    dotnet build $runnerProject -c Release -f $framework --no-restore `
        -p:BuildProjectReferences=false -p:GeneratePackageOnBuild=false `
        -p:CopyLocalRuntimeTargetAssets=false -m:1 -v:q
    if ($LASTEXITCODE -ne 0) { throw "Runner build failed for $framework." }

    $runnerDirectory = "tests/AiDotNet.MgieConditioningReview/bin/Release/$framework"
    Copy-Item -LiteralPath "$tensorReviewRoot/src/AiDotNet.Tensors/bin/Release/$framework/AiDotNet.Tensors.dll" `
        -Destination "$runnerDirectory/AiDotNet.Tensors.dll" -ErrorAction Stop
    Get-FileHash "$runnerDirectory/AiDotNet.dll", "$runnerDirectory/AiDotNetTests.dll", `
        "$runnerDirectory/AiDotNet.Tensors.dll" -Algorithm SHA256

    $priorPath = $env:PATH
    try {
        $nativeDirectory = (Resolve-Path "tests/AiDotNet.MatchaAlignmentReview/bin/Release/$framework").Path
        $env:PATH = "$nativeDirectory;$priorPath"
        $trxName = "pr2136-mgie-final-$framework-$runId.trx"
        dotnet test $runnerProject -c Release -f $framework --no-build --no-restore `
            --logger "trx;LogFileName=$trxName" --results-directory $resultsDirectory
        if ($LASTEXITCODE -ne 0) { throw "Runtime tests failed for $framework." }
        [xml]$report = Get-Content -LiteralPath "$resultsDirectory/$trxName" -Raw
        $counters = $report.TestRun.ResultSummary.Counters
        if ([int]$counters.total -ne 134 -or [int]$counters.passed -ne 134) {
            throw "The complete 134-case cohort did not pass for $framework."
        }
    }
    finally { $env:PATH = $priorPath }
}
```
