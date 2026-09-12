# BLIP-2 and Flamingo ONNX review proof

This follows the BLIP, ImageBind and LLaVA batches on PR #2130. It addresses
the remaining ONNX configuration sites in review threads `ggA2u` and `ggA4B`.
The separate Finch training finding is not covered by this batch.

## Contract and adversarial checks

- Native-only architecture overrides are rejected instead of silently ignored.
  Image geometry is validated independently of an opaque graph's native patch size.
- BLIP-2 checks both sides of its vision/query boundary, including actual runtime
  dimensions when graph axes are symbolic. Its image export must produce the
  configured `NumQueryTokens` and `EmbeddingDimension`; no truncation or zero
  padding hides a mismatch. Vision outputs are kept alive through the query run,
  avoiding the previous full feature clone.
- Required inputs determine which BLIP-2 operations an export supports. Image-only,
  text-only and genuinely defaulted dual-input exports are tested. A required
  joint-input graph cannot be run by either existing operation and is rejected.
  Text-only exports do not invent a visual-query count. Unsupported operations
  fail explicitly before calling an incompatible graph.
- Text tokenization honors the configured context and validates token/mask lengths.
  Existing vector and mean-pooled text embeddings are preserved, including the
  original double-precision pooling, without an intermediate feature copy.
- Flamingo's loaded vision stage returns the graph's real token count and width,
  not an assumed native patch grid. Its native-only perceiver settings are rejected.
- Effective metadata contains immutable graph signatures and validated host
  settings, not guessed native layer counts. Partial constructor failure attempts
  to release every opened session. Production provider selection is unchanged.

These are actual model/ONNX-runtime tests using local data-dependent graphs and
independent scalar sums, means and norms. They are not pretrained quality,
language-generation or GPU-performance proof. Flamingo's two-file ONNX wrapper
still does not load its missing perceiver/generation implementation. BLIP-2's
separate generation-export compatibility is not established by embedding tests.
No generated model-test leaf was manually edited.

## Before / after evidence

The before core is SHA-256
`CC55EBDF00ADCD988787E7AFB2E0BFD4A52092EC17220DF91BC05C0B2D7D7985`.

- Flamingo: 17 failures, zero passes/skips on the previous core.
- BLIP-2 final 30-case cohort: 26 failures, four unchanged positive controls pass,
  zero skips. The controls retain the existing 32-query image export (rank-three
  and single-batch rank-four input) and dynamic vector/token text exports.
- Final before test assembly:
  `ED15A28DFB57C1DA5A5C7EEBA6AFB2BF776BE893074E6DC669AC85E791B28DE1`.
- The first after attempt passed 46/47. Its sole failure was a fixture mistake:
  `AddFloatInitializer` generates a new tensor name, whereas an overridable input
  requires an initializer with the exact input name. The fixture now uses an
  explicit `TensorProto`. Re-running the final fixture against the unchanged
  previous core still gives 26 BLIP-2 failures and four passes. This harness
  correction is not represented as a production defect.

All final runs used .NET 10, CPU, and the same corrected actual AiDotNet assembly:

| Cohort | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| BLIP-2 / Flamingo | 47 | 0 | 0 |
| Existing BLIP / ImageBind / LLaVA | 80 | 0 | 0 |
| Existing CLIP / VideoCLIP | 33 | 0 | 0 |
| Existing GPT-4 Vision | 24 | 0 | 0 |
| Native options / generator regression cohort | 693 | 0 | 0 |
| Total | 877 | 0 | 0 |

The final core build succeeded with zero errors and 2,788 analyzer/existing
warnings (3m43s); the focused runner build had zero warnings/errors. This does
not claim full CI or actual-core replay on the other target frameworks.

- Final actual core:
  `E8FFAB0360580743C254900C40D956C961B4072185D3E7F39EE24E91429515BF`.
- Final query test assembly:
  `ECF8612BD518739052B04ED45EA0D375DD18817CF5067F2E877420D0DC106C97`.
- Final query TRX: `artifacts/pr2130-query-final/results/query-final-after.trx`.
- Regression TRX files in that same directory: `query-regression-CompositeOnnxReview.trx`,
  `query-regression-OnnxOptionsReview.trx`, `query-regression-Gpt4OnnxReview.trx`
  and `query-regression-VisionLanguageOptionsReview.trx`.

Each regression directory contains a copy of this exact corrected core and XML;
previous production/test proof directories were not overwritten. Matching native
ONNX Runtime 1.29.0 assets were present beside the new runner before execution.

Local before evidence is under `artifacts/pr2130-imagebind-llava/results/`:
`flamingo-onnx-before.trx`, `blip2-final-before.trx`; the final before DLL/PDB are
preserved in `artifacts/pr2130-imagebind-llava/query-final-frozen-before/`.
The first intermediate after TRX is
`artifacts/pr2130-query-final/results/query-first-after.trx`.

## Reproduction

From the repository root, build normally so matching native runtime assets are
copied. Local low-disk copy suppression is intentionally not used here.

```powershell
dotnet build tests/AiDotNet.QueryOnnxReview/AiDotNet.QueryOnnxReview.csproj -c Release -f net10.0 -m:1 -p:UseSharedCompilation=false
dotnet test tests/AiDotNet.QueryOnnxReview/AiDotNet.QueryOnnxReview.csproj -c Release -f net10.0 --no-build --no-restore
```

The existing regression projects are `AiDotNet.CompositeOnnxReview`,
`AiDotNet.OnnxOptionsReview`, `AiDotNet.Gpt4OnnxReview` and
`AiDotNet.VisionLanguageOptionsReview` under `tests/`. The native/options cohort
also requires the actual core XML documentation file in its output directory.
