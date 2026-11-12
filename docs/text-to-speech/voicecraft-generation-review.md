# VoiceCraft generation control-flow review

The primary reference is [the author's model and generation loops](https://github.com/jasonppy/VoiceCraft/blob/f0a7a971363905883a21c466a3aa4ea2d4c0b690/models/voicecraft.py). The checkout is clean at this revision. The released 830M checkpoint configuration separately establishes EMPTY 2048, EOG 2049, audio padding 2050 and EOS 2051.

The trained Transformer and codec comparisons did not verify generation control flow. Further adversarial review found these actual mismatches:

- Native repeated-silence handling replaced an already sampled repeated token greedily. The author adjusts the previous silence token's logit before sampling, multiplying negative logits and dividing positive logits by the consecutive-repeat penalty. The counter and threshold also differed.
- Completed codebooks repeatedly fed EOG to the Transformer. The author emits an end marker once per codebook and feeds EMPTY for already completed codebooks while later codebooks finish.
- The original helper terminates when the end marker is the adjusted argmax even if the categorical draw selected another token. Native generation omitted this branch.
- TTS uses EOS when the checkpoint provides it, whereas editing uses EOG. The native shared loop always used EOG. TTS suppresses its end marker at zero-based steps through `encodec_sr // 5`; editing has no such initial-duration gate.
- The author retains all scores tied at the top-k threshold. The native implementation discarded ties to retain exactly k entries. Its top-p comparison also discarded the next entry at an exact cumulative boundary where the author's shifted strict comparison retains it.

`voicecraft_generation_reference.py` extracts the unchanged original top-k/top-p functions and nested editing/TTS helpers by AST, recording the author revision and source SHA-256. Only categorical random draws are supplied by fixed inverse-CDF quantiles. Original filtering, penalties, counters, delayed startup, termination and codebook-flush branches execute unchanged. Six cases cover positive/negative repeated-silence logits, argmax EOG termination, TTS EOS and its initial gate, top-k ties and a top-p boundary. All six fail against production Build47. All six pass against the repaired source, comparing both the recovered codec matrix and every complete delayed row through the flush.

The row comparison additionally exposed the author's early-stop behavior: after flushing begins, pending codebooks are sampled without reapplying startup-delay padding. This was corrected before production verification. Generation now exposes its delayed stream internally and obtains the codec matrix by unshifting it, rather than silently clamping misplaced markers into codec IDs.

Public TTS now passes the EOS mode, while editing passes EOG mode without the TTS minimum-duration or best-of-three selection. Two additional regressions force the actual prediction-head biases and run the real public APIs and EnCodec decoder. They require two generated TTS frames after the configured initial EOS gate and permit an immediate EOG to remove a masked editing frame while retaining the original unmasked frame. Both fail against the previous production assembly. Build48 passes both library targets and all 381 expanded cases in the actual repository test project on each of `net8.0` and `net471`, with zero failures or skips. This includes all eight new generation regressions.

The fixed categorical quantiles compare controlled distributions and do not claim identical Torch/native RNG streams. Fixtures suppress unused special-marker logits; the native sampler continues to constrain acoustic candidates to codec IDs plus the selected end marker. These cases do not certify arbitrary padding-marker distributions or perceptual editing quality. They complement the independent full trained Transformer and custom GigaSpeech codec checks.

Reproduce:

```powershell
python tools/reference-data/voicecraft_generation_reference.py <original-author-checkout>
dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj -f net8.0 --filter 'FullyQualifiedName~VoiceCraftGenerationReferenceTests|FullyQualifiedName~PublicGeneration_UsesDistinct'
```
