# PR2136 aligned token-to-mel review

This runner references the actual AiDotNet library and the shared test initializer.
It does not replace runtime collaborators with a source-linked fake model. Small controlled
fixtures use the actual shared alignment base and actual registered layers.

## Baseline evidence

Production baseline: `2c40f5f30fcf45ae4ba2d52315be28e43c01af6a`.
The production tree was unchanged during the before build and test run.

- Actual net10.0 library and runner build: 0 errors, 2,842 existing warnings.
- Corrected baseline fixture: 2 passed, 2 failed, 0 skipped.
- Both failures show that changing duration width or depth leaves the actual flat parameter
  count unchanged at 380. Both frame-level `[batch, 4]` controls pass (batches 1 and 2).
- TRX: `artifacts/pr2136-matcha-review/pr2136-matcha-before-corrected-fixture.trx`.
- Core SHA-256: `3D98146C840FAD1D59336BBAEF5CDF3DEFEA3DB35A6D5CA5A5398DD4BE12FC08`.
- Test SHA-256: `BEF798B5D8260BA14751CB09774BB7032FDA299BF4DBE7337B23F60E10156B62`.

The earlier `pr2136-matcha-before.trx` has a test-fixture-only missing `inputSize` error.
Its four failures are **not** production-defect evidence. The corrected controls above
were rebuilt against the same unchanged library before running again.

After preserving those binaries, the worktree fast-forwarded to collaborator head
`33da1c31ee0a875f4034419a5d6d571e336d47ef`. Its meta-learning and scaffold changes did
not touch the audio model, LayerHelper, or NeuralNetworkBase; all collaborator commits
are retained.

Before final validation, the additional collaborator range was also inspected and integrated
by fast-forward to `3fc94e7b290f2fc8018c648dd949e9005b516f67`. Its relation-network work and
non-overlapping shared-base clone-mode/Scratch corrections are retained. Only this agent's
shared-base changes were temporarily saved and reapplied, with the recovery stash preserved.

## Intermediate shared-base evidence

The first runtime probes also found defects in the new test harness: the custom model
needed to declare its custom training support, optimizer reevaluation must use the existing
`ReevaluateWithGradients` protocol, and composite-layer assertions must inspect registered
children. Invalid batch validation preserves the caller's prior mode; it does not promise
to switch an untouched model to inference. None of those original assertion failures are
claimed as production defects.

After correcting those oracles, the same actual intermediate library produced 29 passed,
9 failed, 0 skipped in `pr2136-runtime-all-controls-before-base-fix.trx`:

- Additional composite branches were absent from training parameter collection, including
  a model's explicit parameter-selection route. Their dropout mode was not propagated.
- Actual Matcha duration gradients and optimizer updates were absent despite the branch
  being present in the serialized parameter vector.
- Aligned synthesis stayed in training mode, accepted negative-infinite predicted logs,
  and overflowed before dividing two otherwise representable duration/rate values.

Intermediate core SHA-256: `D33BC0EC8FC1C1213223C9C8C7B77CF5F819BC2164031E9D1F0CA6F22251CDFE`.
The exact 38-case before-test SHA-256 is
`CDD693AA977FEEFC2F4541098EDD0F5FD29846EED3BBE4F9DF190FB17D3692A5`.
These are failures in the implementation-in-progress, not nine failures attributed to the
original PR head. The original-head duration-option proof remains the separate two failures above.

After the shared collector/mode and aligned-inference corrections, an actual net10.0
library/runner build completed with 0 errors and 2,783 warnings (3m10s). The same
38 controls passed, with no failures or skips (4s):
`pr2136-shared-composite-after.trx`. This includes real duration-weight updates,
parameter-selection/deduplication, masked gradients, controlled monotonic alignment,
duration-driven mel lengths, and complete parameter/output serialization round-trip.

- After core SHA-256: `29545BDA7A4F8F77F4544FEAFDBB188B02A89A6EC39F2E05E512D899FFA44CB8`.
- After test SHA-256: `F69AD323A0ACB3B706A74B08B15CA236BE738F9F9C3F341614C085E3542CAC16`.

This run deliberately excluded the independently reproduced native vocoder boundary case.
It was not an all-tests-green or GPU execution claim. A separate vocoder before run,
`pr2136-vocoder-contracts-corrected-before.trx`, has 5 passed and 3 failed: the real HiFiGAN
rank-two input crash, missing unbatched normalization, and missing mel-channel validation.
Empty axes were already rejected and are not new production-defect evidence. The earlier
`pr2136-vocoder-contracts-before.trx` incorrectly required an exact base exception type for
two already-valid input-contract rejections; use the corrected run instead.

## Collaborator-integrated runtime evidence

The collaborator-integrated library plus the narrowly gated `VocoderBase` normalization
built with 0 errors and 2,857 warnings (4m54s). All 50 then-existing controls passed with
0 failures and 0 skips (8s), in `pr2136-collaborator-vocoder-final.trx`.

- Core SHA-256: `655E89824B17A18159C82F6B7B75EB918F327FAFD0A2549AA4A01A139300BF17`.
- Exact 50-case test SHA-256: `82645CF8C66E300707E653D735BD7443F23AEFBF598354F152B5FBC321BA7365`.
- Duration depth now changes the actual parameter count from 897 to 1,113; duration width
  changes it from 897 to 1,121. The original-head controls stayed at 380 for both changes.
- One real aligned optimizer step had loss 2.3237004 and changed the 225 duration parameters
  by L2 norm 0.010824375367052903. This is functional gradient/update evidence, not an audio
  quality, convergence-rate, or performance benchmark.
- A caller-owned actual HiFiGAN converts two mel frames to the declared eight waveform
  samples and remains usable after the Matcha model is disposed. Rank-three batch behavior,
  declined step-layout contracts, channel checks, and strided/contiguous gradient controls pass.

An unchanged final test binary also exercised the actual private fused extra-parameter
collector against the preserved intermediate `D33BC0E...` library: it failed because that
collector returned null. With the final library restored and its hash verified, it passed
with the two actual child tensors (five elements). These separate one-case records are
`pr2136-fused-collector-intermediate-before.trx` and
`pr2136-fused-collector-final-after.trx`; they are not GPU execution evidence.

Additional lifecycle checks make this 50-pass snapshot intermediate, not the final handoff.
The corrected six-case run `pr2136-final-lifecycle-retained-before.trx` has four passes and
two failures: synthesis starting in training mode failed to restore that mode after either
success or a non-finite predictor error. The two optimizer partial-write controls passed:
both a fresh writable span and an array retained before cache warmup changed one live
weight from 1 to 5 before throwing; actual subsequent predictions changed from 128 to 132.
The original exception, scheduler success-only behavior, and reusable training sentinel
also passed. No additional cache invalidation change is justified by those passing controls.
This is limited to those actual CPU mutation orderings, not a universal GPU/cache guarantee.
The earlier `pr2136-final-lifecycle-before.trx` included a test-only missing layer
initialization; its apparent cache failure is not production evidence.

The stochastic-mode oracle was also strengthened to use a nonzero actual duration branch:
training dropout can output only 0 or 0.5, while inference must produce exactly 0.25.
Against the unchanged `655E898...` library, those physical assertions and repeatability
pass, but the final caller-mode restoration fails in
`pr2136-mode-stochastic-corrected-before.trx`. The first attempt at this stronger fixture
incorrectly assumed rank-three rather than the actual rank-four Conv1D kernel storage;
`pr2136-mode-stochastic-strengthened-before.trx` is therefore a harness error, not a defect.

Finally, the complete unchanged 56-case suite ran against that same pre-mode-fix library:
53 passed, 3 failed, 0 skipped in `pr2136-final56-exact-before-mode-fix.trx` (9s). All three
failures are caller-mode restoration; both partial-write/cache controls and all other
alignment, training, serialization, and vocoder cases pass. The exact test SHA-256 is
`A8987055B596A506D9CFA625B26C18B75525CF08F66C50182B1305272E27C75A`, preserved as
`artifacts/pr2136-matcha-review/lifecycle-before-AiDotNetTests.dll`. The matching core is
the `655E898...` binary above, also preserved as `final-before-control-AiDotNet.dll`.

## Final lifecycle-corrected validation

The final source restores the caller's training mode in `finally` and performs mel
synthesis without recording an inference graph. The custom-objective helper is unchanged
because its two actual partial-write controls already passed before this correction.

| Target | Actual library and runner build | Focused runtime result | TRX |
| --- | --- | --- | --- |
| net10.0 | 0 errors, 2,789 warnings; 3m42s | 56 passed, 0 failed, 0 skipped; 7s | `pr2136-final56-net10-after.trx` |
| net8.0 | 0 errors, 2,789 warnings; 2m50s | 56 passed, 0 failed, 0 skipped; 6s | `pr2136-final56-net8.0-after.trx` |
| net471 | 0 errors, 2,791 warnings; 2m34s | 56 passed, 0 failed, 0 skipped; 12s | `pr2136-final56-net471-after.trx` |

Final net10.0 core SHA-256:
`A85356AD2042F207CB61D01F749BB7A7C1FFF118B8C4FED54C8BBFB020E5A949`.
Final net10.0 test SHA-256:
`090E4E5A6B1C136D1EF096210901534764865EA0D5032A3236E86339E54B749F`.
The test sources/assertions are unchanged from the exact 56-case before run; the runner
was rebuilt against the updated library, so both distinct binary hashes are recorded.
An independent reviewer replayed the frozen net10.0 binary without rebuilding: all 56
passed with 0 failures and 0 skips (8s), in `pr2136-root-independent-final56.trx`.

Final net8.0 core SHA-256:
`5AA8838F8DCA8CCC802D41AEFE3F1A1DBF82047F64B18AF5B24529E5BBFBF5C8`.
Final net8.0 test SHA-256:
`34D5A9D894DCF916BC4CBA6FCE5F830E415F6AD114CC346169225398E5A3FECE`.

Final net471 core SHA-256:
`BB5A89DC0B4543F05FA42CA6F3CBE391ABE3B72B51684C59E44D768259643C71`.
Final net471 test SHA-256:
`92D233A0E25A1B17B90C4A0F79B0ACCC6C2EAE7BA6B4C6B27BD4A183D8E2A74C`.
All six fixtures explicitly initialize the test engine, including on .NET Framework;
none relies solely on module-initializer execution there. Net10.0 binary hashes were
rechecked after compatibility builds and remain unchanged. No null-forgiving operators
were added in the scoped source/test changes; `git diff --check` is clean.

## Reproduction

Run from the repository root. Keep CPU test policy local to this process; production
continues to use the selected engine, including GPU implementations.

```powershell
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
dotnet restore tests/AiDotNet.MatchaAlignmentReview/AiDotNet.MatchaAlignmentReview.csproj -p:CopyLocalRuntimeTargetAssets=false --disable-parallel
if ($LASTEXITCODE -ne 0) { throw 'Restore failed.' }
$reviewResults = Join-Path 'artifacts/pr2136-matcha-review' ([Guid]::NewGuid().ToString('N'))
foreach ($reviewTarget in @('net10.0', 'net8.0', 'net471')) {
    dotnet build tests/AiDotNet.MatchaAlignmentReview/AiDotNet.MatchaAlignmentReview.csproj -c Release -f $reviewTarget --no-restore -p:GeneratePackageOnBuild=false -p:CopyLocalRuntimeTargetAssets=false -m:1
    if ($LASTEXITCODE -ne 0) { throw "Build failed for $reviewTarget; do not test an older DLL." }
    $reviewTrxName = "pr2136-matcha-$reviewTarget.trx"
    dotnet test tests/AiDotNet.MatchaAlignmentReview/AiDotNet.MatchaAlignmentReview.csproj -c Release -f $reviewTarget --no-build --no-restore --logger "trx;LogFileName=$reviewTrxName" --results-directory $reviewResults
    if ($LASTEXITCODE -ne 0) { throw "Tests failed for $reviewTarget." }
    [xml]$reviewTrx = Get-Content -LiteralPath (Join-Path $reviewResults $reviewTrxName) -Raw
    $reviewCounters = $reviewTrx.TestRun.ResultSummary.Counters
    if ([int]$reviewCounters.executed -ne 56 -or [int]$reviewCounters.passed -ne 56) {
        throw "Expected all 56 cases to execute and pass for $reviewTarget."
    }
}
```

`CopyLocalRuntimeTargetAssets=false` is a test-build-only SDK option that avoids copying
the all-RID native runtime closure. It does not change production package defaults or
prove ONNX-session/GPU native loading.

## Deliberate remaining boundaries

- The frame-level `Predict`/`Train` API and existing constructors remain available.
- New integer-token `SynthesizeMel` returns mels, not a waveform; a caller-owned compatible
  vocoder has a separate explicit boundary.
- Shared native vocoder normalization uses the declared whole-waveform shape contract;
  it does not change arbitrary `IVocoder` implementations or the separate ONNX execution route.
- The existing dense decoder is not a complete conditional-flow-matching U-Net. The new
  alignment/duration path must not be presented as proof of paper-level audio quality.
- The legacy string `Synthesize` tokenizer/prosody/default-waveform behavior is not proved
  correct by these tests and is not silently changed to an unsupported operation.
- MGIE's joint image/instruction conditioning remains a separate unresolved architecture
  task. This batch does not resolve that thread or make the PR ready for merge.

These are actual Windows CPU runs on all three supported target frameworks, not a full
repository-suite run, GPU execution proof, or audio-quality/performance benchmark.
