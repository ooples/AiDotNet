# UniAudio generation length and empty-output review

The primary source is [the original search controller](https://github.com/yangdongchao/UniAudio/blob/c1146a8cecf84f1c2e877170d84998393b7e1e78/UniAudio/infer.py). The released TTS script sets maximum/minimum length ratios to 7 and 0.5. Neural checkpoint comparisons remain separate evidence for global/local logits and codec computations.

`tools/reference-data/uniaudio_generation_reference.py` executes the unchanged original `InferenceImp`, `chunk_after_end` and cache-selection function bodies. A controlled model supplies unique EOS/code logits and empty cache hooks; the original masking, greedy top-k selection, local-codebook loop, length limits, forced-end handling and output truncation execute unchanged. Six cases cover minimum lengths 0, 1, 2, 3, 4 and 6 under a four-frame cap, using three codebooks and nine prefix patches. The fixture records the author revision and source hash.

The original EOS mask applies while `g_idx < minlen - 1`, so minlen 2 permits termination after one complete codec frame. Native synthesis instead used floor(prefix * ratio) as a minimum completed-frame count. It also fabricated a one-frame prompt/zero-code candidate when generation ended without audio tokens. The repair subtracts one when translating the author ratio gate and returns empty audio for an empty candidate. An explicitly requested `MinimumGeneratedFrames` continues to impose that caller's floor; `OfficialV1` no longer adds an unpublished one-frame floor.

`UniAudioGenerationReferenceTests` runs the actual public `SynthesizeWithVoice` path, including text frontend, prefix construction, real global/local transformer and codec decoder. Controlled learned tensors make the final local hidden vector constant and produce the same unique EOS/code maxima as the author controller. Each case verifies the actual nine-patch prefix and returned waveform length against the independent author codec-frame count. A seventh check verifies the released preset's zero additional floor. Six of seven tests fail against the old production assembly; all seven pass against the repaired source.

The reference compares generated token counts. The author's SoundStream detokenizer does not provide a normal waveform decode for an empty sequence. Returning an empty tensor is the native API boundary for that case, not a claim of numerical equality with an original empty decode.

This is a controlled public-pipeline regression, not a trained speech-quality benchmark or certification of identical categorical RNG streams. It does not replace the full released LM and codec checkpoint comparisons.

Build52 compiles the production library on net8.0 and net471 without errors. The normal repository test project passes all 438 reviewed cases on each target, with zero failures or skips. The checkpoint-audit tool also builds without errors.
