# Original CosyVoice checkpoint review

`CosyVoiceClone` inherits `CosyVoice`, with the original 300M defaults. It does not wrap CosyVoice 2. The 300M facade and the canonical CosyVoice 2 model are reviewed separately here. This is component evidence, not signoff on all 18 models.

## Sources and release boundaries

- Original 300M weights: `FunAudioLLM/CosyVoice-300M@24c40509c3c5ea6fe06b5f8790ff99e3714a6bee`. All downloaded files are size- and SHA-256-verified against the repository metadata.
- 300M implementation: author tag `v2.0`, commit `8555549e882236e6541748b1042d95693caa82ba`. Its frontend accepts 16 kHz tensors and resamples prompt features to 22.05 kHz. The newer checkout's legacy frontend hardcodes 24 kHz feature extraction, so using it as a 300M prompt oracle would test a different pipeline.
- CosyVoice 2 weights: `FunAudioLLM/CosyVoice2-0.5B@eec1ae6c79877dbd9379285cf8789c9e0879293d`; implementation `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`. Original source methods and complete trained model states are used.

## Confirmed failures and corrections

The original frozen speech tokenizer and CAM++ are supplied as ONNX graphs, while the native public loader previously required converted frontend tensors. The shared CosyVoice base now supports these original frozen components and retains converted-tensor loading.

The CosyVoice 2 release has 151,663 tokenizer IDs and BlankEN's merge table. The earlier tokenizer used Qwen2.5's added-token layout and different merges. The corrected release tokenizer is checked against the actual author's tokenizer. The older fixture is retained as a correctly labeled generic Qwen2.5 tokenizer check, rather than continuing to call that evidence CosyVoice 2 release parity.

CosyVoice 2's mel frontend must cover its 12 kHz Nyquist limit. It previously inherited the 300M model's 8 kHz cutoff. The explicit override now follows its released configuration. CosyVoice 3 retains its existing Nyquist override.

The 300M vocoder's original file uses legacy `weight_g`/`weight_v` keys. The shared checkpoint reader now accepts these original names when modern parametrization names are absent. Both original PyTorch and safetensors fixtures verify the mapping; malformed sizes and transposed axes are rejected. Full original vocoder verification passes in the source harness.

## Independent numerical evidence

The actual frozen graphs and public `CreateVoice` path pass three recordings per model. For the 300M facade, all 50/63/75 speech codes, transcript IDs and prompt shapes match. Matched double-buffer Matcha mel errors are below 3e-13. For CosyVoice 2, all 25/25/32 codes match and mel errors are below 1.6e-10. CAM++ differences between the original ORT 1.18 execution and the current ORT 1.31 runtime are below 8e-6 with matched frontend precision.

The full 300M LM passes: inputs 2.23e-14, hidden states 1.20e-13, vocabulary logits 6.74e-13, loss 1.92e-13. The full CosyVoice 2 LM passes: inputs exact, logits 1.36e-13, loss 3.56e-15. These strict double comparisons explicitly isolate nonlearned positional buffers and forced float32 casts; original float32 outputs are retained separately.

The complete trained 300M flow passes its conditioned encoder/regulator, initial vector field and all ten prompted Euler steps; final maximum mel error is 2.05e-14. The original neural forward methods are unchanged. Only optional serving imports and a non-neural dtype whitelist are adapted; double operator/buffer precision is explicitly labeled.

The complete original 300M `CosyVoiceClone` source graph also passes through the public checkpoint loader, LM, `TokensToMel`, and `MelToAudio`. All original learned tensors are loaded. Only random draws are replayed from the author execution, including the flow's Gaussian input, harmonic phases, and voiced/unvoiced source noise; no neural forward is substituted. Maximum errors are 6.74e-13 for LM logits, 2.63e-13 for public prompted flow, and 1.79e-13 for the actual stochastic waveform pipeline. This is full 300M inference evidence, not a claim that an independently trained checkpoint was reproduced.

Build39 passes `net8.0` and `net471`, with 330 integrated tests. Build40, including the legacy reader correction, passes both target frameworks and 332 integrated tests. Both frozen frontend audits and the complete original 300M public-facade checkpoint audit above now also pass against the Build40 production assembly, with the same numerical bounds as the source harness. The later independent FastSpeech graph checks add two passing tests without changing production sources.
