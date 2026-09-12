# ImageBind / LLaVA ONNX contract proof

This closes the ImageBind and LLaVA portions of review thread `ggA3Q`; GPT-4
Vision's previously pushed proof is in `PR2130_GPT4_ONNX_REVIEW_PROOF.md`.

## Changes and boundaries

- ONNX constructors reject native-only architecture overrides they cannot honor.
  Host image/context/embedding dimensions are validated against loaded graph
  signatures, separately from native patch/layer construction requirements.
- Immutable `OnnxConfiguration` describes the actual graph inputs/outputs and
  host dimensions. ONNX metadata no longer reports guessed native internals.
- ImageBind validates all three encoders, checks actual waveform feature length
  against fixed audio inputs, and rejects wrong-width or multi-batch outputs
  instead of truncating them. Its existing normalized first-token behavior is
  preserved for token-feature outputs.
- LLaVA converts its single-batch graph features into the two-dimensional
  sequence consumed by its existing mean pool. Token count comes from the graph,
  not an assumed patch size. Symbolic counts remain unknown: `NumVisualTokens`
  then throws a documented exception, while execution returns the actual count.
  Its two-graph wrapper has no separate ONNX projector, so the vision export must
  already produce the configured language-input width.
- Partial constructor failure attempts to dispose every opened session. Native
  model construction/forward paths and production provider selection are unchanged.

This is not pretrained quality, generation, or GPU performance proof. ImageBind's
pre-existing audio preprocessing is an energy-feature approximation, not a real
mel transform. LLaVA's pre-existing ONNX text-embedding/generation limitations are
not repaired by these embedding-boundary changes. The PR remains draft for its
other unfinished review findings.

## Actual before / after

The new tests execute local, data-dependent ONNX graphs through the actual
AiDotNet assembly. Numerical checks use independent scalar input sums, feature
offsets, mean pooling, and norms; there is no replacement model implementation.

| Selected cohort | Before | After | Skipped |
| --- | --- | --- | ---: |
| ImageBind expanded cases | 1 passed, 32 failed | 33 passed | 0 |
| LLaVA expanded cases | 0 passed, 20 failed | 20 passed | 0 |
| Existing BLIP ONNX cases | Previously 27 passed | 27 passed | 0 |
| Existing CLIP / VideoCLIP ONNX | Previously 33 passed | 33 passed | 0 |
| Existing GPT-4 Vision ONNX | Previously 24 passed | 24 passed | 0 |
| Native/options/generator regression cohort | Previously 693 passed | 693 passed | 0 |

All after checks ran on .NET 10, CPU, against the same corrected production DLL.
The actual core build succeeded with zero errors and 2,850 existing/analyzer
warnings; the focused test build had zero warnings/errors. These counts do not
claim full CI or an actual-core replay on the other target frameworks.

Some before failures are missing diagnostics/configuration, not incorrect
arithmetic: ImageBind's positive embedding arithmetic already passed before
the metadata assertion, and its dynamic-duration control passed outright.
LLaVA's single-vector arithmetic also worked before its missing-configuration
assertion. The tests separately demonstrate wrong shape, discarded batch,
incorrect token-count reporting, and silent truncation failures.

The first combined after attempt was aborted during ONNX Runtime native API
initialization (`0xC0000005`) after nine passing cases. This low-disk build had
suppressed native runtime copies. The managed DLL was version 1.29.0; placing the
matching 1.29.0 native runtime/provider DLLs beside the test assembly resolved the
crash. The unchanged production/test DLLs then passed all 80 combined cases.
Both the aborted and completed TRX/log files are retained; the abort is not
presented as a model regression or as a successful run.

## Reproduce

From the repository root, build normally so the matching native assets are
copied (the local low-disk copy suppression is intentionally not used here):

```powershell
dotnet build tests/AiDotNet.CompositeOnnxReview/AiDotNet.CompositeOnnxReview.csproj -c Release -f net10.0 -m:1 -p:UseSharedCompilation=false
dotnet test tests/AiDotNet.CompositeOnnxReview/AiDotNet.CompositeOnnxReview.csproj -c Release -f net10.0 --no-build --no-restore
```

The other focused projects are `AiDotNet.OnnxOptionsReview`,
`AiDotNet.Gpt4OnnxReview`, and `AiDotNet.VisionLanguageOptionsReview` under `tests/`.
Build and test them with the same framework/configuration. The final native
cohort's runner includes the corrected core XML documentation file.

Local evidence:

- Before: `artifacts/pr2130-onnx-contracts/results/imagebind-onnx-expanded-before.trx`
  and `llava-onnx-expanded-before.trx` in the same directory.
- After: `artifacts/pr2130-imagebind-llava/results/imagebind-llava-blip-native-closure-after.trx`
  and the three `imagebind-llava-regression-*.trx` files in the same directory.
- Frozen before test DLL/PDB: `artifacts/pr2130-imagebind-llava/frozen-before/`.

SHA-256:

- Before core: `BC40C3928D37C803726ED4D1297069D062E6013C0B40408BFB435C4C46C26BEC`.
- Expanded combined before tests: `E008D244EC16BA2BC87908D2735A0174DA8F10DD1EF7C13BF45FBA69B78C1468`.
- After core: `CC55EBDF00ADCD988787E7AFB2E0BFD4A52092EC17220DF91BC05C0B2D7D7985`.
- After combined tests: `F1DEB29381ABC6554902BBA2B4239DD997AB8584C84315315B8E986B0DAC0EF9`.
- Matching native ONNX Runtime: `69D8E6D3879A3B4001CDC74C8ED9CCC7E7F799A5B847059738323404519EC471`.
