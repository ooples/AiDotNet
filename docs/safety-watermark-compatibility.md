# Safety watermark migration from 0.234.0

The keyed watermark implementation merged in #2413 replaces heuristic watermark scores with actual embedding and statistical detection. Existing marks and custom implementations need migration before upgrading. The package comparison continues to reject unrelated API breaks.

Ten original constructor signatures are retained as forwarding overloads, including protected base constructors. They use the current default configuration. Existing compiled calls still resolve, but retaining a signature does not promise identical watermark algorithms or scores.

The remaining baseline suppressions cover these specific changes on net471, net8.0 and net10.0:

| Change | Reason and migration |
| --- | --- |
| `TextWatermarker(byte[], double, double, int)` | The keyed generator now accepts a string key and the generator's tokenizer. Construct it with an application secret and the actual tokenizer, and regenerate marks. A byte-array adapter would not make the old marks compatible and would make explicit null-key calls ambiguous with the string overload. |
| New abstract text `Detect` and watermark description | Implement the actual statistical test and return its z-score, p-value and scored-unit count. The old confidence heuristic cannot provide these measurements. |
| New abstract audio frame embedding, detection weights, pattern salt and description | Custom subclasses must implement the keyed spectral algorithm. A default no-op would silently omit a requested watermark. |
| New interface `Detect` and audio `EmbedWatermark` members | Custom implementations must provide the new detection result and real embedding operation. Recompile consumers after migrating implementations. |
| `AudioWatermarkerBase.DetectWatermark` becomes concrete | The shared implementation maps the real detection result to confidence; existing overrides remain valid. |
| `WatermarkDetector` no longer implements the generated parameter registrar | The composite detector holds keyed statistical detectors rather than trainable model parameters. Code that treats it as a trainable parameter registrar must stop doing so. |

Each suppression names one diagnostic and one exact member or type, is restricted to the released-package baseline, and retains framework-specific paths. No diagnostic category is disabled globally. Constructor availability and the keyed text/audio round trips, false-positive calibration, perturbation tolerance and CLIP alignment behavior are covered by regression tests.

Manual validation: all 62 focused safety cases passed without skips. The library compiled for net10.0, net8.0, net471, net462 and netstandard2.0, and `dotnet pack --no-build` passed package validation against 0.234.0 with all five assets. These local builds disabled analyzers; they do not replace the separate analyzer gate.
