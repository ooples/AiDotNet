# Chatterbox inference-policy review

The primary source is [the author's T3 inference loop](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/src/chatterbox/models/t3/t3.py), the same clean revision used by the full trained T3/voice-encoder checkpoint reference. Its `pyproject.toml` pins Transformers 5.2.0, matching the reference environment. The checkpoint comparisons remain separate evidence for learned logits and conditioning.

`chatterbox_sampling_reference.py` extracts the actual inference-loop statements for CFG, repetition penalties, temperature, min-p, top-p and final softmax, together with the original processor construction statements. They execute unchanged against controlled conditional/unconditional logits. The fixture records the author revision, T3 source SHA-256, exact Transformers version and installed logit-policy source SHA-256. Fixed inverse-CDF quantiles compare distributions without assuming identical Torch/native RNG algorithms.

Six initial cases agree with the old native implementation, covering non-boundary CFG, positive/negative repetition penalties, temperature, min-p and nucleus composition. Two exact-boundary cases fail: native descending cumulative `> p` filtering retains an extra token, whereas the author's `TopPLogitsWarper` removes the ascending tail whose cumulative probability is `<= 1-p`, retaining at least one token. A ninth case shows that valid `top_p=0` selects an argmax under the author policy; native code incorrectly disabled nucleus filtering for zero.

The native implementation now follows the ascending-tail rule and handles zero consistently with the pinned policy. Three of nine comparisons fail against the previous production assembly; all nine pass against the repaired source, over 54 fixed categorical quantiles and both sides of each resolved active-distribution CDF boundary at `1e-9`. The controlled vectors include equal probabilities to expose the exact boundaries. Their deterministic CPU tie order is part of these fixtures; arbitrary tied outputs across GPU libraries and RNG implementations are not claimed identical.

An additional integration regression runs the actual reduced T3 transformer with a zero speech-head matrix. For its eight-token configuration, the author policy's one-eighth nucleus contains only configured stop ID 7. Old production emits three tokens through the real generation loop instead of terminating; the repaired source returns no acoustic tokens. This verifies that the generation loop uses the corrected sampler. It is a controlled-parameter integration test, not a claim that the released 8,194-row head has the same uniform-logit outcome.

The public release defaults retain `top_p=1.0`; the repair affects exposed non-default nucleus settings and their boundaries. The original default CFG/repetition/temperature/min-p processing order is unchanged. The reference vectors use float64; full trained float32/operator-precision differences retain their separately reported checkpoint tolerances.

All ten cases pass in the source harness and the rebuilt production assembly. Build50 passes both library frameworks, the checkpoint-audit executable build, and all 421 reviewed cases through the actual repository test project on each of `net8.0` and `net471`, with zero failures or skips. The negative repeated-logit vector was subsequently strengthened to retain its entire distribution rather than prune it to an argmax; all ten cases pass again through each actual test-project target, including the full active CDF boundary probes. Copied fixture hashes match the current repository data on both targets. The checked author T3 source and committed dependency file have no tracked modifications.

Reproduce:

```powershell
python tools/reference-data/chatterbox_sampling_reference.py <pinned-chatterbox-checkout>
dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj -f net8.0 --filter FullyQualifiedName~ChatterboxSamplingReferenceTests
```
