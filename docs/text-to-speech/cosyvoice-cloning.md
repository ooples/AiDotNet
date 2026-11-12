# CosyVoice voice cloning

Use `AiDotNet.TextToSpeech.CodecBased.CosyVoice<T>` and `CosyVoiceOptions` for CosyVoice synthesis and cloning.
The duplicate `VoiceCloning.CosyVoiceClone<T>` and `CosyVoiceCloneOptions` have been removed. Their native synthesis
generated a fixed-frequency waveform rather than running the released model's networks.

```csharp
using AiDotNet.TextToSpeech.CodecBased;

using var model = new CosyVoice<float>(architecture, new CosyVoiceOptions());
model.Voice = model.CreateVoice(referenceAudio, referenceSampleRate, "the reference recording's transcript");
var waveform = model.Synthesize("Hello from CosyVoice.");
```

The native path uses the S3 speech tokenizer and CAM++ speaker encoder to create the prompt, the speech-token
language model, the conditional flow-matching network, and the HiFT vocoder. Omit the transcript for cross-lingual
prompting. The voice's speaker embedding is the CAM++ x-vector used by the model, rather than statistics over
waveform samples. Model defaults generate 22.05 kHz audio.

Train through `TtsTrainingSample<T>` and `CurrentStage`, supplying the supervision required by each stage. The
language model, flow and vocoder train separately; S3 and CAM++ remain frozen as in the released recipe.
Train or load the native weights before evaluating speech quality. The ONNX constructor remains available.
