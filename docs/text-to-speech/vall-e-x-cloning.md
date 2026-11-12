# VALL-E X voice cloning

Use `AiDotNet.TextToSpeech.CodecBased.VALLEX<T>` and `VALLEXOptions` for cross-lingual voice cloning.
The duplicate `VoiceCloning.VALLEXClone<T>` and `VALLEXCloneOptions` have been removed. Their native synthesis
generated a fixed-frequency waveform instead of running VALL-E X's trained models.

```csharp
using AiDotNet.TextToSpeech.CodecBased;

using var model = new VALLEX<float>(architecture, new VALLEXOptions());
model.Voice = model.CreateVoice(referenceAudio24kHz, "the reference recording's transcript");
var waveform = model.Synthesize("你好，世界。");
```

The reference recording supplies EnCodec prompt tokens; its transcript supplies phoneme tokens. VALL-E X uses
these prompts and language IDs in its autoregressive and non-autoregressive models. It does not extract a separate
speaker embedding. For training, use `TtsTrainingSample<T>` with codec tokens, the prompt-sentence codes and
the utterance's language ID. `CurrentStage` selects the autoregressive or non-autoregressive objective.

Paper defaults construct the full architecture. Native weights must be trained or loaded before evaluating speech
quality. The constructor accepting an ONNX path remains available for exported models.
