# PR #2130: additional GAN and vision-language documentation review

## Scope and baseline

This follow-up addresses exactly four assigned review threads on
[PR #2130](https://github.com/ooples/AiDotNet/pull/2130), starting at
`e1c1335cf84dabb106de5f7d8a96a1480e010fc4`.
Expected remote branch: `feature/options-surface-phase-3-vislang`.
All four comment lists were fetched completely; every `hasNextPage` was false.

| Thread | Finding and result |
| --- | --- |
| `PRRT_kwDOKSXUF86gf_-n` / [3964555442](https://github.com/ooples/AiDotNet/pull/2130#discussion_r3964555442) | The shared GAN base now requires both channel counts to be positive. |
| `PRRT_kwDOKSXUF86gf_-z` / [3964555465](https://github.com/ooples/AiDotNet/pull/2130#discussion_r3964555465) | Six VideoCLIP properties document units, actual effects, implementation defaults and native/ONNX distinctions. |
| `PRRT_kwDOKSXUF86ggA27` / [3964560814](https://github.com/ooples/AiDotNet/pull/2130#discussion_r3964560814) | Both BLIP constructors and the CLIP constructor document their actual options parameter, without orphaned scalar tags. |
| `PRRT_kwDOKSXUF86ggA3V` / [3964560846](https://github.com/ooples/AiDotNet/pull/2130#discussion_r3964560846) | AVC sampling units and the explicitly named BLIP/BLIP2/LLaVA/ImageBind numeric-property cohort now have substantive documentation. |

There are no ONNX behavior changes, public API renames, gradient-policy changes,
generated scaffold edits, new skips or tolerance changes. The only executable
production change is two `Require` calls in `GanOptions.ValidateCore`.

## Adversarial consumer checks

The current repository has **no production subclass or consumer of GanOptions**.
BigGAN and SAGAN use separate scalar constructor arguments and private fields;
their capitalized channel-name occurrences are metadata keys. Therefore these
tests prove the protected public-base validation contract, **not a fix to an
existing GAN model's construction path**. No concrete GAN defaults or new channel
fallbacks were invented. The abstract base documents that concrete consumers must
supply required dimensions and that it does not replace existing model APIs.

The rate-documentation suggestions also needed correction:

- VideoCLIP reads `FrameRate` into an exposed metadata field. Actual frame selection
  calls `SampleFrames(frames, NumFrames)`, sampling uniformly and repeating the
  last supplied frame when necessary. The rate does not resample input or change
  playback speed.
- AVC `VideoFrameRate` is also metadata. `GetVisualEmbedding` averages the supplied
  frames rather than selecting them by rate.
- AVC `AudioSampleRate` actively configures the cached log-mel front end; changing
  it does not itself resample a waveform.
- ImageBind native audio capacity uses rate and duration; its audio-encoding
  method still has a separate input sample-rate argument. IMU counts observations,
  not milliseconds. The ONNX video path uses the first supplied frame.
- Native layer widths/depths are explicitly distinguished from loaded ONNX graphs;
  documentation does not promise that ignored ONNX architecture options are honored.

Default provenance was checked against the actual pre-migration constructors at
`671836a347436f4b023ebe3f02a86ea333ad0686` (`ad22ea9cf1^`), not inferred from
model names or papers:

| Options | Preserved implementation values |
| --- | --- |
| VideoCLIP | Frames 8; rate 1.0 frames/s; text width 512; frame/temporal/text blocks 12/4/12 |
| AVC | Audio rate 16000 samples/s; video rate 25 frames/s (former named constants) |
| LLaVA | Language-model blocks 32 |
| BLIP | Decoder blocks 12; feed-forward width 3072 |
| BLIP2 | Query/language widths 768/2560; query blocks 12; query tokens 32; decoder blocks 6 |
| ImageBind | Audio rate 16000 samples/s; duration 10 seconds; IMU observations 2000; video frames 2 |

These are implementation provenance claims, not newly verified paper-default claims.

## Failure-first and final evidence

The small runner source-links the actual options, their complete real base chain,
and their real enums. The new unit tests are also included normally by the main
test project; no replacement production classes or hand-edited generated tests
are involved.

| Run | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Original source, new 34-case cohort, net10 | 6 | 28 | 0 |
| Fixed source, complete scalar suite, net10 | 384 | 0 | 0 |
| Fixed source, complete scalar suite, net8 | 384 | 0 | 0 |
| Fixed source, complete scalar suite, net471 | 384 | 0 | 0 |

The 28 baseline failures are six nonpositive-channel cases, 20 missing emitted
property-documentation cases and two absent rate-metadata disclosures. The six
baseline passes are positive channel and existing-required-field controls.
Every invalid channel test validates an otherwise valid probe first and requires
an exception naming both `options` and the exact property. Zero critic iterations
remain valid. Positive scalar channel counts do not claim model allocation safety
at arbitrarily large sizes.

The 20 documentation cases inspect compiler-emitted XML and the real constructor
values. The final complete suite contains 350 pre-existing and 34 new cases,
repeated on each framework. All scalar builds succeeded without reported compiler
warnings/errors; test execution was approximately 125/93/374 milliseconds.
No new warning suppression was added.

A separate Roslyn constructor/signature audit found these original XML mismatches:

- BLIP ONNX: three orphaned tags and missing `options`.
- BLIP native: eleven orphaned tags and missing `options`.
- CLIP: three orphaned tags and missing `options`.

After correction all three constructor blocks contain exactly their declared
parameters, without duplicates. Roslyn `SyntaxFactory.AreEquivalent`, ignoring
trivia, also proves all eight documentation-only files preserve their executable
syntax relative to the baseline.

## Outputs, reproduction and limits

All new build and intermediate outputs were redirected into
`artifacts/pr2130-additional-review/build`. Existing main/native binaries,
the earlier native proof, and its fixture source/case lists were not changed.
The six existing main DLL hashes still match the earlier
[PR2130 proof](PR2130_REVIEW_PROOF.md) after this work.

The actual library/model families were **not rebuilt or rerun in this batch**.
That is deliberate: only the unconsumed shared scalar GAN base changes behavior.
The earlier native-model proof remains its earlier evidence, not a claim that
these new scalar tests ran inside the existing main binaries. Fresh main CI and
independent review remain required before broad merge-readiness claims.

Results are retained in `artifacts/pr2130-additional-review/results/`:

- `additional-reviews-baseline.trx`
- `additional-reviews-final-net10.0.trx`
- `additional-reviews-final-net8.0.trx`
- `additional-reviews-final-net471.trx`

Final source-linked test-assembly SHA-256:

| Framework | SHA-256 |
| --- | --- |
| net10.0 | `1A333D8DA335F30B377FE29F64429CC1D4B684C52D928CD74140FFDF7BFCFA6A` |
| net8.0 | `84D0EDB25F6236FF20A79825F0BDDD12E490CDAF9B042958B2CF31F70015A865` |
| net471 | `A998F1AB074EACDE819FE0172A598EBE7FC2F8FB43D31E574185080AA7A591F7` |

Run from the repository root on Windows, preserving the output redirection:

```powershell
$ErrorActionPreference = 'Stop'
foreach ($framework in @('net10.0', 'net8.0', 'net471')) {
    dotnet test tests/AiDotNet.OptionsContractTests/AiDotNet.OptionsContractTests.csproj -c Release -f $framework --artifacts-path artifacts/pr2130-additional-review/reproduction-build -m:1 -p:UseSharedCompilation=false --logger "trx;LogFileName=additional-reviews-reproduction-$framework.trx" --results-directory artifacts/pr2130-additional-review/reproduction-results -v:quiet
    if ($LASTEXITCODE -ne 0) { throw "Scalar build or $framework tests failed." }
}
```

The failure-first run used the same redirected build path, only the new-test
filter `FullyQualifiedName~GanOptionsContractTests|FullyQualifiedName~VisionLanguageOptionsDocumentationTests`,
and the unchanged baseline production files. No pushes, review resolutions or
merge actions were performed as part of this local proof.

